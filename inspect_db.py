import sqlite3

def main():
    db_path = "card_capture_output/cards.sqlite"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
        
    print("\n--- Saved Cards ---")
    cursor.execute("SELECT * FROM saved_cards WHERE video_id = 4")
    for row in cursor.fetchall():
        print(dict(row))

if __name__ == '__main__':
    main()
