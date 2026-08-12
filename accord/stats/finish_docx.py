"""
Finish the Word build.

Pandoc discards \\newpage when converting LaTeX to docx, so every chapter and
every front-matter page runs on from the previous one. The PDF is correct
because LaTeX honours the breaks; the Word file needs them inserted afterwards.

This inserts a real Word page break before every top-level heading, so each
chapter, appendix and front-matter section starts on its own page, and leaves
everything else untouched.

Run:  python3 accord/stats/finish_docx.py IN.docx OUT.docx
"""

import re
import shutil
import sys
import zipfile

PAGEBREAK = ('<w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>'
             '<w:r><w:br w:type="page"/></w:r></w:p>')

# Paragraph styles that should start a new page.
TOP_LEVEL = ("Heading1", "TOCHeading")

# Front-matter pages. Pandoc renders \section* as Heading2, so they cannot be
# separated by style alone; they are matched by their text instead. Each of
# these must begin its own page under the Morgan State format.
FRONT_MATTER = ("Declaration of Authorship", "ABSTRACT", "Acknowledgements",
                "List of Abbreviations")


def add_page_breaks(xml):
    """Insert a page break before each top-level heading except the first."""
    paras = re.findall(r"<w:p\b.*?</w:p>", xml, re.S)
    inserted = 0
    seen_first = False
    for p in paras:
        m = re.search(r'<w:pStyle w:val="([^"]+)"', p)
        style = m.group(1) if m else ""
        text = re.sub(r"<[^>]+>", "", p).strip()
        is_front = any(text.startswith(f) for f in FRONT_MATTER)
        if style not in TOP_LEVEL and not is_front:
            continue
        if not seen_first:            # the first heading opens the document
            seen_first = True
            continue
        xml = xml.replace(p, PAGEBREAK + p, 1)
        inserted += 1
    return xml, inserted


def main():
    src, dst = sys.argv[1], sys.argv[2]
    shutil.copy(src, dst)
    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}

    xml = blobs["word/document.xml"].decode("utf8")
    xml, n = add_page_breaks(xml)
    blobs["word/document.xml"] = xml.encode("utf8")

    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.writestr(name, blobs[name])

    print(f"  inserted {n} page breaks")
    print(f"  wrote {dst}")


if __name__ == "__main__":
    main()
