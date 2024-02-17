versions=("AKJV" "AMP" "ASV" "BBE" "BSB" "DBY" "ESV" "GEN" "KJV" "MSG" "NASB" "TWI" "UKJV" "WBT" "WEB" "YLT")

sqlite3 Bible_Database.db <<EOF
CREATE TABLE IF NOT EXISTS bible (
    version TEXT NOT NULL,
    book INT NOT NULL,
    chapter INT NOT NULL,
    verse INT NOT NULL,
    text TEXT
)
EOF

sqlite3 Bible_Database.db <<EOF
DELETE FROM bible
EOF

for version in "${versions[@]}"
do
    echo "Importing ${version}"

    sqlite3 Bible_Database.db <<EOF
ATTACH '${version}Bible_Database.db' AS source;
INSERT INTO Bible (version, book, chapter, verse, text)
SELECT '${version}', Book, Chapter, Versecount, verse FROM source.bible;
DETACH source;
EOF

    echo "Finished importing ${version}"
done
