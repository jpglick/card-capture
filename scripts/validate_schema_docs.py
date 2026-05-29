import re
import sys
from pathlib import Path

def extract_sql_schema(migrations_dir: Path) -> dict[str, set[str]]:
    schema = {}
    for sql_file in sorted(migrations_dir.glob("*.sql")):
        content = sql_file.read_text()
        # Very simple parser: look for CREATE TABLE IF NOT EXISTS table_name (
        # Then grab columns until );
        table_blocks = re.findall(r"CREATE TABLE.*?([a-zA-Z0-9_]+)\s*\((.*?)\);", content, re.DOTALL | re.IGNORECASE)
        for table_name, block in table_blocks:
            table_name = table_name.lower()
            if table_name not in schema:
                schema[table_name] = set()
            
            lines = block.split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('--') or line.startswith('UNIQUE') or line.startswith('PRIMARY KEY') or line.startswith('FOREIGN KEY'):
                    continue
                # Column name is the first word
                parts = line.split()
                if parts:
                    col_name = parts[0].lower()
                    schema[table_name].add(col_name)
                    
        # Look for ALTER TABLE table_name ADD COLUMN col_name ...
        alter_blocks = re.findall(r"ALTER TABLE\s+([a-zA-Z0-9_]+)\s+ADD COLUMN\s+([a-zA-Z0-9_]+)", content, re.IGNORECASE)
        for table_name, col_name in alter_blocks:
            table_name = table_name.lower()
            if table_name in schema:
                schema[table_name].add(col_name.lower())
                
    return schema

def extract_md_schema(md_file: Path) -> dict[str, set[str]]:
    content = md_file.read_text()
    schema = {}
    
    # Very simple parser: look for CREATE TABLE IF NOT EXISTS table_name (
    # Then grab columns until );
    table_blocks = re.findall(r"CREATE TABLE.*?([a-zA-Z0-9_]+)\s*\((.*?)\);", content, re.DOTALL | re.IGNORECASE)
    for table_name, block in table_blocks:
        table_name = table_name.lower()
        if table_name not in schema:
            schema[table_name] = set()
        
        lines = block.split('\n')
        for line in lines:
            line = line.split('--')[0] # Remove comments
            line = line.strip()
            if not line or line.startswith('UNIQUE') or line.startswith('PRIMARY KEY') or line.startswith('FOREIGN KEY'):
                continue
            # Column name is the first word
            parts = line.split()
            if parts:
                col_name = parts[0].lower()
                schema[table_name].add(col_name)
                
    # Look for ALTER TABLE table_name ADD COLUMN col_name ...
    alter_blocks = re.findall(r"ALTER TABLE\s+([a-zA-Z0-9_]+)\s+ADD COLUMN\s+([a-zA-Z0-9_]+)", content, re.IGNORECASE)
    for table_name, col_name in alter_blocks:
        table_name = table_name.lower()
        if table_name in schema:
            schema[table_name].add(col_name.lower())
            
    return schema

def main():
    root_dir = Path(__file__).parent.parent
    sql_schema = extract_sql_schema(root_dir / "migrations")
    md_schema = extract_md_schema(root_dir / "docs" / "contracts" / "storage-schema.md")
    
    has_error = False
    
    for table, cols in sql_schema.items():
        if table in ("_migrations", "fb_labels_new"):
            continue
        if table not in md_schema:
            print(f"ERROR: Table '{table}' found in SQL but missing from docs.")
            has_error = True
        else:
            missing_in_md = cols - md_schema[table]
            if missing_in_md:
                print(f"ERROR: Columns {missing_in_md} in table '{table}' found in SQL but missing from docs.")
                has_error = True
                
    for table, cols in md_schema.items():
        if table not in sql_schema:
            print(f"ERROR: Table '{table}' found in docs but missing from SQL.")
            has_error = True
        else:
            missing_in_sql = cols - sql_schema[table]
            if missing_in_sql:
                print(f"ERROR: Columns {missing_in_sql} in table '{table}' found in docs but missing from SQL.")
                has_error = True
                
    if has_error:
        sys.exit(1)
    else:
        print("Schema validation passed.")
        sys.exit(0)

if __name__ == "__main__":
    main()
