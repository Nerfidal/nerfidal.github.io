import calendar
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

API_URL = "http://export.arxiv.org/api/"

MONTHS = dict(zip(range(1, 13), list(calendar.month_name)[1:]))

SEP = {"author": "<author:sep>", "comment": "<comment:sep>", "tag": "<tag:sep>"}

METADATA_TEMPLATE = """---
layout: default
title: {title}
parent: {parent}
nav_order: {order}
---

<!---metadata--->

"""

TAG_TEMPLATE = """{tag}
{{: .label .label-blue }}
"""

CARD_TEMPLATE = """
## {title}

{tags}
| Published | Authors | Category | |
|:---:|:---:|:---:|:---:|
| {pub_date} | {authors} | {category} | [PDF]({pdf_link}){{: .btn .btn-green }} |

**Abstract**: {abstract}

{comments}
"""


class ArxivAgent:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.database = pd.read_csv(self.db_path, sep=";").fillna(
            value={"tags": "", "comments": ""}
        )
        self.xml: BeautifulSoup = None

    def update_database(self, new_data):
        # removing old version papers
        new_ids = new_data.id.str.split("v").str[0]
        db = self.database[~self.database.id.str.split("v").str[0].isin(new_ids)]
        # adding new papers
        self.database = (
            pd.concat([new_data, db])
            .sort_values("pub_date", ascending=False)
            .reset_index(drop=True)
        )
        self.database.to_csv(self.db_path, sep=";", index=False)

    def get_query_url(self, tags: list[str], num_results: int = 10):
        search_query = "+OR+".join(f'all:"{t}"' for t in tags)
        url = (
            f"query?search_query={search_query}"
            "&sortBy=submittedDate"
            "&sortOrder=descending"
            f"&max_results={num_results}"
        )
        return API_URL + url

    def parse_paper_entry(self, paper: BeautifulSoup, tags: list[str]):
        title = paper.title.text.strip().replace("\n", "")
        abstract = paper.summary.text.strip()
        link = paper.id.text
        return {
            "id": link.rsplit("/", 1)[-1],
            "link": link,
            "pub_date": paper.published.text.split("T")[0],
            "title": title,
            "abstract": abstract,
            "authors": SEP["author"].join(
                a.find("name").text for a in paper.find_all("author")
            ),
            "pdf_link": paper.find("link", {"type": "application/pdf"})["href"],
            "category": paper.find("arxiv:primary_category")["term"],
            "comments": SEP["comment"].join(
                [c.text for c in paper.find_all("arxiv:comment")]
            ),
            "tags": SEP["tag"].join(t for t in tags if t in (title + abstract).lower()),
        }

    def filter_new_papers(self, new_papers_df):
        new_df = new_papers_df[~new_papers_df.id.isin(self.database.id)]
        return new_df if len(new_df) > 0 else None

    def call_api(self, tags: list[str], num_results: int = 10):
        url = self.get_query_url(tags, num_results)
        response = requests.get(url)
        response.raise_for_status()

        self.xml = BeautifulSoup(response.text, features="xml")

    def generate_page(self, year: str, month: str):
        papers = self.database[
            self.database.pub_date.str.startswith(f"{year}-{month:02d}")
        ].to_dict("records")

        cards = [
            CARD_TEMPLATE.format(
                title=p["title"],
                tags="\n".join(
                    TAG_TEMPLATE.format(tag=t)
                    for t in p["tags"].split("<tag:sep>")
                    if t
                ),
                pub_date=p["pub_date"],
                authors=p["authors"].replace("<author:sep>", ", "),
                category=p["category"],
                pdf_link=p["pdf_link"],
                abstract=p["abstract"],
                comments="Comments:\n{}".format(
                    "\n".join(f"- {c}" for c in p["comments"].split("<comment:sep>"))
                )
                if p["comments"]
                else "",
            )
            for p in papers
        ]
        metadata = METADATA_TEMPLATE.format(
            title=f"{MONTHS[month]} {year}", parent="Papers", order=year * 100 + month
        )
        return metadata + "\n---\n".join(cards)

    def generate_pages(self, months_to_generate: list[str]):
        papers_path = self.db_path.parent / "docs" / "papers"
        for year_month in months_to_generate:
            year, month = year_month.split("-")
            page = self.generate_page(int(year), int(month))

            page_path = papers_path / (year + month + ".md")
            page_path.write_text(page)

    def main(
        self,
        tags: list[str] = ["gaussian splatting", "nerf"],
        num_results: int = 10,
        generate_pages: bool = True,
    ):
        self.call_api(tags, num_results)
        paper_entries = self.xml.find_all("entry")
        papers = [self.parse_paper_entry(p, tags) for p in paper_entries]
        papers = pd.DataFrame(papers).sort_values("pub_date", ascending=False)

        new_papers = self.filter_new_papers(papers)
        if new_papers is not None:
            self.update_database(new_papers)
            print("{} new paper(s) found.".format(len(new_papers)))
            if generate_pages:
                months_to_update = (
                    new_papers.pub_date.str.rsplit("-", n=1).str[0].unique()
                )
                self.generate_pages(months_to_update)
        else:
            print("No new papers were found.")


if __name__ == "__main__":
    agent = ArxivAgent("papers.csv")
    agent.main(tags=["gaussian splatting", "nerf"], num_results=10, generate_pages=True)
