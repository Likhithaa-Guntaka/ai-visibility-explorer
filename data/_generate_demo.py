"""Generate the SYNTHETIC demo dataset for AI Visibility Explorer.

Run once to (re)create ``demo_prompts.csv`` and ``demo_responses.csv``::

    python data/_generate_demo.py

IMPORTANT — these responses are 100% SYNTHETIC. They were written by this script,
not produced by ChatGPT, Claude, Gemini, Perplexity, or any real AI platform. The
``platform`` column uses labels like "ChatGPT (synthetic)" to make that explicit.
The data is designed to be *realistic enough to exercise every metric* while being
honestly fictional. Nothing here should be presented as a real measurement.

The generator is fully deterministic (fixed random seed) so the CSVs are stable and
the test suite / README findings stay reproducible.
"""

from __future__ import annotations

import csv
import hashlib
import os
import random

random.seed(42)  # deterministic output


def _stable_hash(*parts) -> int:
    """Deterministic, cross-process hash (Python's built-in hash() is randomized)."""
    joined = "|".join(str(p) for p in parts)
    return int(hashlib.md5(joined.encode("utf-8")).hexdigest(), 16)

HERE = os.path.dirname(os.path.abspath(__file__))

BRANDS = ["Notion", "Asana", "ClickUp", "Monday.com", "Trello"]

# A rough, invented "ground truth" popularity profile so the metrics tell a story.
# Higher weight => more likely to be mentioned and mentioned earlier / recommended.
BRAND_STRENGTH = {
    "Notion": 0.92,
    "Asana": 0.80,
    "ClickUp": 0.62,
    "Monday.com": 0.58,
    "Trello": 0.40,
}

# Realistic-looking third-party source domains that AI answers commonly lean on,
# plus each brand's own domain. All fictional in this context.
THIRD_PARTY_SOURCES = [
    ("https://www.g2.com/categories/project-management", "g2.com"),
    ("https://www.capterra.com/project-management-software/", "capterra.com"),
    ("https://zapier.com/blog/best-project-management-software/", "zapier.com"),
    ("https://www.pcmag.com/picks/the-best-project-management-software", "pcmag.com"),
    ("https://www.nytimes.com/wirecutter/reviews/best-to-do-list-app/", "nytimes.com"),
    ("https://www.reddit.com/r/productivity/comments/best-pm-tool/", "reddit.com"),
    ("https://www.forbes.com/advisor/business/software/best-project-management-software/", "forbes.com"),
]

BRAND_DOMAINS = {
    "Notion": ("https://www.notion.so/product", "notion.so"),
    "Asana": ("https://asana.com/product", "asana.com"),
    "ClickUp": ("https://clickup.com/features", "clickup.com"),
    "Monday.com": ("https://monday.com/product", "monday.com"),
    "Trello": ("https://trello.com/tour", "trello.com"),
}

# ---------------------------------------------------------------------------
# Prompt set: 22 prompts spanning all seven required categories.
# Columns: prompt_id, prompt_text, prompt_category, topic, persona,
#          journey_stage, is_brand_prompt
# ---------------------------------------------------------------------------
PROMPTS = [
    # Informational
    ("p01", "What is project management software and how does it help teams?", "Informational", "Category basics", "Team Lead", "Awareness", False),
    ("p02", "How do productivity apps help remote teams stay organized?", "Informational", "Remote work", "Remote Manager", "Awareness", False),
    ("p03", "What features should a good task management tool have?", "Informational", "Features", "Operations Manager", "Awareness", False),
    # Nonbrand discovery
    ("p04", "What are the best project management tools in 2026?", "Nonbrand discovery", "Best tools", "Team Lead", "Consideration", False),
    ("p05", "Recommend a productivity app for a small startup team.", "Nonbrand discovery", "Startup tools", "Startup Founder", "Consideration", False),
    ("p06", "Which tools do agencies use to manage client projects?", "Nonbrand discovery", "Agency workflow", "Agency Owner", "Consideration", False),
    ("p07", "What software helps teams track tasks and deadlines?", "Nonbrand discovery", "Task tracking", "Operations Manager", "Consideration", False),
    # Product comparison
    ("p08", "Compare Notion vs Asana for team collaboration.", "Product comparison", "Notion vs Asana", "Team Lead", "Consideration", True),
    ("p09", "ClickUp vs Monday.com: which is better for project tracking?", "Product comparison", "ClickUp vs Monday", "Operations Manager", "Consideration", True),
    ("p10", "Trello vs Asana for a small marketing team, which should I pick?", "Product comparison", "Trello vs Asana", "Marketing Manager", "Consideration", True),
    ("p11", "How does Notion compare to traditional project management tools?", "Product comparison", "Notion positioning", "Team Lead", "Consideration", True),
    # Purchase intent
    ("p12", "Which project management tool has the best free plan for startups?", "Purchase intent", "Free plans", "Startup Founder", "Decision", False),
    ("p13", "What is the most affordable project management software for a 10-person team?", "Purchase intent", "Pricing", "Operations Manager", "Decision", False),
    ("p14", "I'm ready to buy a task management tool for my agency, what do you recommend?", "Purchase intent", "Agency purchase", "Agency Owner", "Decision", False),
    # Problem based
    ("p15", "My team keeps missing deadlines. What software can help us stay on track?", "Problem based", "Missed deadlines", "Team Lead", "Consideration", False),
    ("p16", "We use too many disconnected apps. What tool can consolidate our workflow?", "Problem based", "Tool sprawl", "Operations Manager", "Consideration", False),
    ("p17", "How can a remote team improve visibility into who is working on what?", "Problem based", "Work visibility", "Remote Manager", "Consideration", False),
    # Customer persona
    ("p18", "What is the best productivity tool for a solo freelancer?", "Customer persona", "Freelancer", "Freelancer", "Consideration", False),
    ("p19", "What project management app works best for a non-technical marketing team?", "Customer persona", "Marketing team", "Marketing Manager", "Consideration", False),
    ("p20", "What tool should a fast-growing engineering team use to plan sprints?", "Customer persona", "Engineering team", "Engineering Lead", "Consideration", False),
    # Brand specific
    ("p21", "Is Notion good for managing complex team projects?", "Brand specific", "Notion capability", "Team Lead", "Decision", True),
    ("p22", "What are the main limitations of Trello for growing teams?", "Brand specific", "Trello limitations", "Operations Manager", "Decision", True),
]

PLATFORMS = [
    ("ChatGPT (synthetic)", "gpt-synthetic-demo"),
    ("Claude (synthetic)", "claude-synthetic-demo"),
    ("Gemini (synthetic)", "gemini-synthetic-demo"),
]

RUN_DATE = "2026-07-10"


def _ordered_brands_for_prompt(prompt_id: str, salt: int) -> list[str]:
    """Pick and order the brands mentioned in one synthetic answer.

    Each brand is *probabilistically* included based on its strength, so mention
    rates differ realistically across brands (strong brands appear often but not
    always; weaker brands appear sometimes). Included brands are then ordered by
    strength plus per-run jitter, so repeated runs mostly agree but occasionally
    reorder — the noise the consistency metric is meant to surface.
    """
    rng = random.Random(_stable_hash(prompt_id, salt) & 0xFFFFFFFF)
    included = []
    for b in BRANDS:
        # Inclusion probability tracks strength but leaves headroom below 1.0 so even
        # the strongest brand is occasionally absent.
        p_include = 0.35 + 0.55 * BRAND_STRENGTH[b]
        if rng.random() < p_include:
            included.append(b)
    # Guarantee at least two brands so comparisons make sense.
    if len(included) < 2:
        ranked = sorted(BRANDS, key=lambda b: BRAND_STRENGTH[b] + rng.uniform(-0.1, 0.1), reverse=True)
        included = ranked[:2]
    included.sort(key=lambda b: BRAND_STRENGTH[b] + rng.uniform(-0.2, 0.2), reverse=True)
    return included


# Two descriptor variants per brand. The variant is chosen by PLATFORM so different
# platforms describe the same brand slightly differently — this deliberately creates
# realistic narrative variation and a few genuine conflicts (e.g. "affordable" vs
# "pricey") for the Entity & Narrative Analysis page to surface.
BLURB_VARIANTS = {
    "Notion": [
        "**Notion** is an all-in-one workspace that combines docs, wikis, databases, and project boards. Its strength is flexibility and a generous free plan, though it can be overwhelming to set up for beginners.",
        "**Notion** is a flexible all-in-one workspace with docs, wikis, and databases. It is simple to start with ready-made templates and is popular with startups and small teams.",
    ],
    "Asana": [
        "**Asana** is a mature project management platform known for reliable task tracking, timelines, and workload management. It is powerful for larger teams, but its pricing gets expensive at scale.",
        "**Asana** offers polished task tracking, timelines, and reporting. It is affordable for small teams and is often recommended for marketing teams.",
    ],
    "ClickUp": [
        "**ClickUp** is a highly customizable, all-in-one tool bundling tasks, docs, goals, and automations at an affordable price, though the interface can feel cluttered and the learning curve is steep.",
        "**ClickUp** packs powerful automations, dashboards, and goals into one platform. It is feature-rich, but it can get pricey on higher tiers.",
    ],
    "Monday.com": [
        "**Monday.com** offers colorful, spreadsheet-style boards, automations, and dashboards that non-technical teams find approachable and easy to use, but advanced reporting is limited on lower tiers.",
        "**Monday.com** provides visual boards with powerful dashboards and reporting. Its automations suit larger teams, though it can feel expensive as you add seats.",
    ],
    "Trello": [
        "**Trello** uses a simple, intuitive Kanban-board approach that is easy to learn and has a free plan, though it feels limited for complex projects and lacks advanced reporting.",
        "**Trello** is a lightweight Kanban tool that is easy to learn and great for simple workflows, with a free plan that appeals to freelancers and small teams.",
    ],
}


def _blurb_for(brand: str, platform: str) -> str:
    """Deterministically pick a brand's descriptor variant based on the platform."""
    variants = BLURB_VARIANTS[brand]
    return variants[_stable_hash(brand, platform) % len(variants)]


def _build_response_text(prompt_id: str, ordered: list[str], with_citations: bool, platform: str, rng: random.Random) -> str:
    """Compose a synthetic answer that mentions brands in order and may cite sources."""
    intro = rng.choice(
        [
            "Here are some strong options to consider:",
            "Based on common recommendations, these tools stand out:",
            "A few widely used tools fit this need:",
            "Several popular platforms could work well here:",
        ]
    )
    lines = [intro, ""]
    for i, b in enumerate(ordered, start=1):
        lines.append(f"{i}. {_blurb_for(b, platform)}")
    lines.append("")
    # The top-ranked brand is framed as the recommendation.
    lines.append(f"For most teams, **{ordered[0]}** is a great starting point, with **{ordered[1]}** as a solid alternative.")

    if with_citations:
        lines.append("")
        lines.append("Sources:")
        # Always a couple of third-party sources, plus the winner's own domain.
        picks = rng.sample(THIRD_PARTY_SOURCES, k=rng.choice([2, 2, 3]))
        for url, _dom in picks:
            lines.append(f"- {url}")
        own_url, _own_dom = BRAND_DOMAINS[ordered[0]]
        lines.append(f"- {own_url}")
    return "\n".join(lines)


def build() -> None:
    """Write the two demo CSV files to the data directory."""
    prompt_rows = []
    for pid, text, cat, topic, persona, stage, is_brand in PROMPTS:
        prompt_rows.append(
            {
                "prompt_id": pid,
                "project_id": "demo",
                "prompt_text": text,
                "prompt_category": cat,
                "topic": topic,
                "persona": persona,
                "journey_stage": stage,
                "is_brand_prompt": str(is_brand).lower(),
            }
        )

    response_rows = []
    run_counter = 1
    for pid, *_ in PROMPTS:
        # Each prompt gets answered on 2 platforms; a subset gets a repeated run
        # (run_number 2) on one platform so consistency can be measured.
        chosen_platforms = random.sample(PLATFORMS, k=2)
        for platform, model in chosen_platforms:
            n_runs = 2 if random.random() < 0.5 else 1
            for run_number in range(1, n_runs + 1):
                ordered = _ordered_brands_for_prompt(pid, salt=run_counter)
                # ~75% of answers include citations.
                with_citations = random.random() < 0.75
                rng = random.Random(_stable_hash(pid, platform, run_number) & 0xFFFFFFFF)
                text = _build_response_text(pid, ordered, with_citations, platform, rng)
                response_rows.append(
                    {
                        "run_id": f"r{run_counter:03d}",
                        "prompt_id": pid,
                        "platform": platform,
                        "model_name": model,
                        "run_date": RUN_DATE,
                        "run_number": run_number,
                        "response_text": text,
                        "has_citations": str(with_citations).lower(),
                        "dataset_kind": "Synthetic",
                        "benchmark_name": "Demo Synthetic Benchmark",
                        "collection_date": RUN_DATE,
                        "collection_notes": "Script-generated synthetic response (not a real AI platform output).",
                    }
                )
                run_counter += 1

    _write_csv(
        os.path.join(HERE, "demo_prompts.csv"),
        ["prompt_id", "project_id", "prompt_text", "prompt_category", "topic", "persona", "journey_stage", "is_brand_prompt"],
        prompt_rows,
    )
    _write_csv(
        os.path.join(HERE, "demo_responses.csv"),
        ["run_id", "prompt_id", "platform", "model_name", "run_date", "run_number", "response_text", "has_citations",
         "dataset_kind", "benchmark_name", "collection_date", "collection_notes"],
        response_rows,
    )
    print(f"Wrote {len(prompt_rows)} prompts and {len(response_rows)} synthetic responses.")


def _write_csv(path: str, header: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    build()
