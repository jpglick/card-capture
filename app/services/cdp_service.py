"""CardDealerPro integration service.

Provides submit (single + bulk) and status-poll operations backed by the
`carddealerpro` Python package.  Authentication uses CDP_EMAIL / CDP_PASSWORD
env vars; the library caches the token to ~/.carddealerpro/token.json so
subsequent calls are instant.

Lifecycle for a card after upload:
  1. upload_card()              → status = 'submitted'  (draft in batch)
  2. CDP processes async        → poll until is_processed
  3. apply_suggested_prices_bulk() → writes start_price on the CDP card
  4. add_cards_to_inventory()   → moves card into inventory (appears in listing)
  5. Our DB: status = 'identified'

Steps 3+4 are run automatically whenever we detect is_processed=True during any
poll.  Without them cards stay in the batch as drafts and never appear in CDP's
inventory view.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, List, Optional

log = logging.getLogger(__name__)


def _get_client():
    """Return an authenticated carddealerpro Client.

    Tries the on-disk token cache first; falls back to a fresh login using
    CDP_EMAIL / CDP_PASSWORD environment variables.
    """
    from carddealerpro.client import Client

    cached = Client.from_cache()
    if cached is not None:
        return cached
    return Client.login()


def _finalize_identified_cards(client, cards) -> None:
    """Apply suggested prices and add identified cards to inventory.

    This is the missing step: CDP cards stay as drafts in a batch until
    (a) their start_price is written via the bulk-update endpoint, and
    (b) they are explicitly POSTed to /inventory/add-cards.

    Only cards where is_identified=True are eligible; cards that processed
    but failed to match (is_processed but not is_identified) are skipped.
    """
    from carddealerpro.pricing import apply_suggested_prices_bulk
    from carddealerpro.inventory import add_cards_to_inventory

    identified = [c for c in cards if c.is_identified]
    if not identified:
        return

    try:
        result = apply_suggested_prices_bulk(client, identified)
        log.info("CDP: applied prices to %d/%d cards", result.get("applied", 0), len(identified))
    except Exception as exc:
        log.warning("CDP: price apply failed (non-fatal): %s", exc)

    ids = [c.id for c in identified]
    try:
        add_cards_to_inventory(client, ids)
        log.info("CDP: added %d card(s) to inventory", len(ids))
    except Exception as exc:
        log.error("CDP: add_to_inventory failed: %s", exc)
        raise


class CdpService:
    """Service layer for CardDealerPro submissions."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal DB helpers
    # ------------------------------------------------------------------

    def _conn(self):
        from card_capture.data.connection import read_connection
        return read_connection(str(self.db_path))

    def _write(self, sql: str, params: tuple = ()) -> None:
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(sql, params)
            conn.commit()

    # ------------------------------------------------------------------
    # Resolve the best image path for a card
    # ------------------------------------------------------------------

    def _resolve_image(self, instance_id: int) -> Optional[Path]:
        """Return the best available image file for a card instance."""
        from card_capture.data.sql_queries import CARD_GET_ONE, CARD_CANONICAL_RECTIFIED

        with self._conn() as conn:
            row = conn.execute(CARD_GET_ONE, (instance_id,)).fetchone()
            if not row:
                return None
            # row: (id, track_id, video_id, run_id, angle, fused_path, created_at)
            fused = row[5]

        out_root = Path(os.environ.get("CC_OUTPUT") or "var/output")

        def _try(stored: Optional[str]) -> Optional[Path]:
            if not stored:
                return None
            p = Path(stored)
            if p.exists():
                return p
            # Try relative to output root
            for part_i in range(len(p.parts) - 1):
                if p.parts[part_i] == "var" and p.parts[part_i + 1] == "output":
                    tail = "/".join(p.parts[part_i + 2:])
                    cand = out_root / tail
                    if cand.exists():
                        return cand
            return None

        img = _try(fused)
        if img:
            return img

        # Fallback: canonical rectified
        with self._conn() as conn:
            canon = conn.execute(CARD_CANONICAL_RECTIFIED, (instance_id,)).fetchone()
        if canon:
            img = _try(canon[0])
        return img

    # ------------------------------------------------------------------
    # Shared card-state classifier
    # ------------------------------------------------------------------

    def _classify_card(self, cdp_card) -> tuple[str, Optional[str], Optional[float], Optional[str]]:
        """Return (new_status, identified_name, suggested_price, raw_json) for a CDP card."""
        if cdp_card.is_identified:
            identified_name = (
                cdp_card.raw.get("name")
                or cdp_card.raw.get("card_name")
                or cdp_card.raw.get("title")
            )
            return (
                "identified",
                identified_name,
                cdp_card.suggested_price,
                json.dumps(cdp_card.raw),
            )
        elif cdp_card.has_errors:
            return "failed", None, None, json.dumps(cdp_card.raw)
        elif cdp_card.is_processed:
            # Processed but unmatched — no market data / no match
            return "identified", None, None, json.dumps(cdp_card.raw)
        else:
            return "processing", None, None, None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_submission(self, instance_id: int) -> Optional[dict[str, Any]]:
        """Return the CDP submission record for a card, or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT cdp_batch_id, cdp_card_id, status, identified_name, "
                "suggested_price, submitted_at, updated_at "
                "FROM cdp_submissions WHERE card_instance_id = ?",
                (instance_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "cdp_batch_id": row[0],
            "cdp_card_id": row[1],
            "status": row[2],
            "identified_name": row[3],
            "suggested_price": row[4],
            "submitted_at": row[5],
            "updated_at": row[6],
        }

    def get_run_submissions(self, run_id: str) -> dict[str, dict[str, Any]]:
        """Return a dict of instance_id -> submission for all cards in a run."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT s.card_instance_id, s.cdp_batch_id, s.cdp_card_id,
                       s.status, s.identified_name, s.suggested_price,
                       s.submitted_at, s.updated_at
                FROM cdp_submissions s
                JOIN card_instances ci ON ci.id = s.card_instance_id
                WHERE ci.run_id = ? AND ci.hidden = 0
                """,
                (run_id,),
            ).fetchall()
        result = {}
        for r in rows:
            result[str(r[0])] = {
                "cdp_batch_id": r[1],
                "cdp_card_id": r[2],
                "status": r[3],
                "identified_name": r[4],
                "suggested_price": r[5],
                "submitted_at": r[6],
                "updated_at": r[7],
            }
        return result

    def submit_card(self, instance_id: int, batch_name: Optional[str] = None) -> dict[str, Any]:
        """Submit a single card to CardDealerPro.

        Creates a new batch named after the card if batch_name is not given.
        Returns the submission record.
        """
        img = self._resolve_image(instance_id)
        if img is None:
            raise LookupError(f"No image file found for card instance {instance_id}")

        client = _get_client()

        from carddealerpro.batches import create_batch, upload_card

        name = batch_name or f"card-capture-{instance_id}"
        batch = create_batch(client, name)
        card = upload_card(client, batch.id, img)

        cdp_card_id = str(card.id) if card.id is not None else None

        self._write(
            """
            INSERT INTO cdp_submissions
                (card_instance_id, cdp_batch_id, cdp_card_id, status)
            VALUES (?, ?, ?, 'submitted')
            ON CONFLICT(card_instance_id) DO UPDATE SET
                cdp_batch_id = excluded.cdp_batch_id,
                cdp_card_id  = excluded.cdp_card_id,
                status       = 'submitted',
                updated_at   = datetime('now')
            """,
            (instance_id, str(batch.id), cdp_card_id),
        )
        log.info("Submitted card %s to CDP batch %s (card_id=%s)", instance_id, batch.id, cdp_card_id)
        return self.get_submission(instance_id)

    def bulk_submit_run(self, run_id: str) -> dict[str, Any]:
        """Submit all visible (non-hidden) cards in a run to a single CDP batch.

        Returns a summary: {batch_id, submitted, skipped, failed}.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT ci.id FROM card_instances ci
                WHERE ci.run_id = ? AND ci.hidden = 0
                ORDER BY ci.id
                """,
                (run_id,),
            ).fetchall()
        instance_ids = [r[0] for r in rows]

        if not instance_ids:
            return {"batch_id": None, "submitted": 0, "skipped": 0, "failed": 0, "errors": []}

        # Check which are already submitted — skip those
        with self._conn() as conn:
            existing = {
                r[0]
                for r in conn.execute(
                    "SELECT card_instance_id FROM cdp_submissions WHERE card_instance_id IN ({})".format(
                        ",".join("?" * len(instance_ids))
                    ),
                    instance_ids,
                ).fetchall()
            }

        to_submit = [i for i in instance_ids if i not in existing]

        if not to_submit:
            return {"batch_id": None, "submitted": 0, "skipped": len(instance_ids), "failed": 0, "errors": []}

        client = _get_client()
        from carddealerpro.batches import create_batch, upload_card

        batch_name = f"card-capture-run-{run_id[:8]}"
        batch = create_batch(client, batch_name)
        batch_id = str(batch.id)

        submitted = 0
        failed = 0
        errors: list[str] = []

        for iid in to_submit:
            img = self._resolve_image(iid)
            if img is None:
                log.warning("No image for card %s — skipping CDP upload", iid)
                errors.append(f"instance {iid}: no image file")
                failed += 1
                continue
            try:
                card = upload_card(client, batch.id, img)
                cdp_card_id = str(card.id) if card.id is not None else None
                self._write(
                    """
                    INSERT INTO cdp_submissions
                        (card_instance_id, cdp_batch_id, cdp_card_id, status)
                    VALUES (?, ?, ?, 'submitted')
                    ON CONFLICT(card_instance_id) DO UPDATE SET
                        cdp_batch_id = excluded.cdp_batch_id,
                        cdp_card_id  = excluded.cdp_card_id,
                        status       = 'submitted',
                        updated_at   = datetime('now')
                    """,
                    (iid, batch_id, cdp_card_id),
                )
                submitted += 1
            except Exception as exc:
                log.error("Failed to submit card %s to CDP: %s", iid, exc)
                errors.append(f"instance {iid}: {exc}")
                failed += 1

        log.info(
            "Bulk submit run %s: batch=%s submitted=%d failed=%d skipped=%d",
            run_id, batch_id, submitted, failed, len(instance_ids) - len(to_submit),
        )
        return {
            "batch_id": batch_id,
            "submitted": submitted,
            "skipped": len(instance_ids) - len(to_submit),
            "failed": failed,
            "errors": errors,
        }

    def poll_submission(self, instance_id: int) -> dict[str, Any]:
        """Poll CDP for the latest status of a submitted card and update the DB.

        When a card is identified for the first time, automatically applies the
        suggested price and adds it to the CDP inventory so it appears in listings.

        Returns the updated submission record.
        """
        sub = self.get_submission(instance_id)
        if sub is None:
            raise LookupError(f"No CDP submission for card instance {instance_id}")

        if sub["status"] in ("identified", "failed"):
            return sub  # Terminal — no need to re-poll

        client = _get_client()
        from carddealerpro.batches import get_batch_cards
        from carddealerpro.models import Card as CdpCard

        cards = get_batch_cards(client, sub["cdp_batch_id"])
        cdp_card_id = sub.get("cdp_card_id")

        matched: Optional[CdpCard] = None
        if cdp_card_id:
            for c in cards:
                if str(c.id) == str(cdp_card_id):
                    matched = c
                    break
        if matched is None and cards:
            matched = cards[0]

        if matched is None:
            return sub

        new_status, identified_name, suggested_price, raw = self._classify_card(matched)

        # Finalize: apply price + move to inventory when newly identified
        if new_status == "identified" and matched.is_identified:
            try:
                _finalize_identified_cards(client, [matched])
            except Exception as exc:
                log.error("CDP finalize failed for card %s: %s", instance_id, exc)

        self._write(
            """
            UPDATE cdp_submissions
            SET status = ?, identified_name = ?, suggested_price = ?,
                raw_response = ?, updated_at = datetime('now')
            WHERE card_instance_id = ?
            """,
            (new_status, identified_name, suggested_price, raw, instance_id),
        )
        return self.get_submission(instance_id)

    def poll_batch(self, batch_id: str) -> list[dict[str, Any]]:
        """Poll all cards in a CDP batch and update their DB rows.

        Applies suggested prices and adds newly-identified cards to the CDP
        inventory in one bulk call so they appear in the inventory listing.

        Returns list of updated submission records.
        """
        client = _get_client()
        from carddealerpro.batches import get_batch_cards

        cards = get_batch_cards(client, batch_id)
        card_map = {str(c.id): c for c in cards if c.id is not None}

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, card_instance_id, cdp_card_id, status FROM cdp_submissions WHERE cdp_batch_id = ?",
                (batch_id,),
            ).fetchall()

        # Separate newly-identified cards (not yet in terminal state in our DB)
        # so we can finalize them in one batch call instead of one-by-one.
        newly_identified_cdp_cards = []

        updated = []
        for (sub_id, instance_id, cdp_card_id, current_status) in rows:
            cdp_card = card_map.get(str(cdp_card_id)) if cdp_card_id else None
            if cdp_card is None:
                continue

            new_status, identified_name, suggested_price, raw = self._classify_card(cdp_card)

            # Collect cards that just became identified (weren't terminal before)
            if new_status == "identified" and current_status not in ("identified", "failed"):
                if cdp_card.is_identified:
                    newly_identified_cdp_cards.append(cdp_card)

            self._write(
                """
                UPDATE cdp_submissions
                SET status = ?, identified_name = ?, suggested_price = ?,
                    raw_response = ?, updated_at = datetime('now')
                WHERE card_instance_id = ?
                """,
                (new_status, identified_name, suggested_price, raw, instance_id),
            )
            updated.append(self.get_submission(instance_id))

        # Finalize all newly-identified cards in one round-trip
        if newly_identified_cdp_cards:
            try:
                _finalize_identified_cards(client, newly_identified_cdp_cards)
            except Exception as exc:
                log.error("CDP finalize failed for batch %s: %s", batch_id, exc)

        return updated
