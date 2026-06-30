#!/usr/bin/env python3
"""Query the arXiv API and print structured results."""
import sys, time, urllib.parse
import requests
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}

def search(query, max_results=12, start=0):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    out = []
    for e in root.findall("a:entry", NS):
        aid = e.find("a:id", NS).text.strip()
        title = " ".join(e.find("a:title", NS).text.split())
        summary = " ".join(e.find("a:summary", NS).text.split())
        authors = [a.find("a:name", NS).text for a in e.findall("a:author", NS)]
        published = e.find("a:published", NS).text[:10]
        cats = [c.attrib.get("term") for c in e.findall("a:category", NS)]
        out.append({"id": aid, "title": title, "authors": authors,
                    "published": published, "summary": summary, "cats": cats})
    return out

if __name__ == "__main__":
    q = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    for i, p in enumerate(search(q, n)):
        short = p["id"].split("/abs/")[-1]
        print(f"\n[{i}] {short}  ({p['published']})  {','.join(p['cats'][:3])}")
        print(f"    TITLE: {p['title']}")
        print(f"    AUTHORS: {', '.join(p['authors'][:6])}")
        print(f"    ABSTRACT: {p['summary'][:700]}")
