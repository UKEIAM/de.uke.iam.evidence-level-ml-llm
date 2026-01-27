import requests
import csv
from pathlib import Path
import yaml

GRAPHQL_URL = "https://civicdb.org/api/graphql"

# GraphQL query template
query_template = """
query ($first: Int!, $after: String) {
  evidenceItems(first: $first, after: $after) {
    nodes {
      id
      status
      molecularProfile {
        id
        name
      }
      disease {
        id
        name
      }
      therapies {
        id
        name
      }
      variantOrigin
      significance
      evidenceLevel
      evidenceType
      description
      source {
        id
        sourceType
        citation
        citationId
        title
        abstract
      }
    }
    pageInfo {
      endCursor
      hasNextPage
    }
  }
}
"""


def main():
    with open("config/data_config.yaml", "r") as f:
        data_config = yaml.safe_load(f)
        print("Fetching CIViC evidence items...")

        # Pagination loop
        all_items = []
        first = 5000
        after = None

        while True:
            variables = {"first": first, "after": after}
            response = requests.post(
                GRAPHQL_URL, json={"query": query_template, "variables": variables}
            )
            response.raise_for_status()
            data = response.json()

            nodes = data["data"]["evidenceItems"]["nodes"]
            page_info = data["data"]["evidenceItems"]["pageInfo"]

            all_items.extend(nodes)
            print(f"Fetched {len(nodes)} items, total so far: {len(all_items)}")

            if not page_info["hasNextPage"]:
                break
            after = page_info["endCursor"]

        print(f"Finished fetching all evidence items. Total: {len(all_items)}")

        # Flatten items for CSV
        def flatten(item):
            return {
                "id": item.get("id"),
                "status": item.get("status"),
                "evidence_type": item.get("evidenceType"),
                "molecular_profile": item["molecularProfile"]["name"]
                if item.get("molecularProfile")
                else "",
                "molecular_profile_id": item["molecularProfile"]["id"]
                if item.get("molecularProfile")
                else "",
                "disease": item["disease"]["name"] if item.get("disease") else "",
                "disease_id": item["disease"]["id"] if item.get("disease") else "",
                "therapies": ",".join([t["name"] for t in item.get("therapies", [])])
                if item.get("therapies")
                else "",
                "therapies_id": ",".join(
                    [str(t["id"]) for t in item.get("therapies", [])]
                )
                if item.get("therapies")
                else "",
                "variant_origin": item.get("variantOrigin"),
                "significance": item.get("significance"),
                "evidence_level": item.get("evidenceLevel"),
                "description": item.get("description"),
                "source_id": item["source"]["id"] if item.get("source") else "",
                "source_type": item["source"]["sourceType"]
                if item.get("source")
                else "",
                "citation": item["source"]["citation"] if item.get("source") else "",
                "pubmed_id": item["source"]["citationId"] if item.get("source") else "",
                "title": item["source"]["title"] if item.get("source") else "",
                "abstract": item["source"]["abstract"] if item.get("source") else "",
            }

        # Ensure output directory exists
        output_dir = Path(data_config["raw_dir_path"])
        output_dir.mkdir(parents=True, exist_ok=True)

        flat_items = [flatten(i) for i in all_items]

        # Write to CSV
        fieldnames = list(flat_items[0].keys())

        with open(
            output_dir / data_config["raw_csv_filename"],
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_items)

        print("Saved CSV")


if __name__ == "__main__":
    main()
