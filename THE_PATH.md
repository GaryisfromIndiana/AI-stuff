# THE PATH — Empire AI Roadmap

## Layer 0: Engine Validation (1-2 sessions)
**Goal**: Prove the ACE pipeline works end-to-end.

- [ ] Single lieutenant executes a research task via Planner -> Executor -> Critic
- [ ] Task produces real output stored in memory and knowledge graph
- [ ] Critic loop catches and fixes at least one issue
- [ ] Evolution cycle runs: propose -> review -> apply
- [ ] Verify: memory retrieval returns stored findings

**Success metric**: One lieutenant completes a research task autonomously with verified storage.

---

## Layer 1: Data Pipeline Smoke Test (1-2 sessions)
**Goal**: Verify all data sources return real data.

- [ ] Web search returns results and stores them
- [ ] arXiv search returns papers with abstracts
- [ ] GitHub search returns repos with metadata
- [ ] HuggingFace search returns models/datasets
- [ ] Reddit search returns posts with scores
- [ ] Hacker News search returns stories
- [ ] Semantic Scholar returns papers with citations
- [ ] Papers With Code returns trending papers
- [ ] URL scraping extracts content and stores it
- [ ] News search returns current articles

**Success metric**: All 10 data sources work. Entity extraction produces KG entries.

---

## Layer 2: Knowledge Accumulation (2-3 sessions)
**Goal**: Build up the knowledge base autonomously.

- [ ] AutoResearcher runs a full cycle: gaps -> questions -> search -> extract -> synthesize
- [ ] Strategy tracker records outcomes and picks better strategies over time
- [ ] Knowledge graph has 100+ entities across all lieutenant domains
- [ ] Memory has synthesis reports from at least 3 domains
- [ ] Gap detection identifies real gaps (not just "everything")
- [ ] Novelty filter prevents duplicate storage

**Success metric**: Empire has enough knowledge to produce informed analysis without external prompting.

---

## Layer 3: Output Channel Setup (2-3 sessions)
**Goal**: Connect Empire to the outside world.

- [ ] Twitter/X bot posts research summaries
- [ ] Blog generator creates markdown posts from synthesis reports
- [ ] Discord bot shares findings in a channel
- [ ] GitHub Pages site publishes research reports
- [ ] RSS feed of latest findings

**Success metric**: At least 2 output channels publishing automatically.

---

## Layer 4: Content Pipeline (2-3 sessions)
**Goal**: Automate research-to-content.

- [ ] Research findings auto-generate tweet threads
- [ ] Weekly synthesis reports auto-publish to blog
- [ ] Trending paper summaries posted within 24 hours
- [ ] War Room debates produce publishable analysis
- [ ] Content quality gate (Critic reviews before publishing)

**Success metric**: Daily content output with no manual intervention.

---

## Layer 5: Autonomous Loop (2-3 sessions)
**Goal**: Close the loop — research -> publish -> measure -> adapt.

- [ ] Scheduler drives research cycles on cadence (every 4-6 hours)
- [ ] Output metrics tracked (engagement, reach)
- [ ] Research priorities adjust based on what gets engagement
- [ ] Budget tracking prevents runaway costs
- [ ] Health monitoring alerts on failures

**Success metric**: Empire runs for 48+ hours producing useful output without intervention.

---

## Layer 6: Quality & Freshness (2-3 sessions)
**Goal**: Make output trustworthy and current.

- [ ] Source credibility scoring weights findings
- [ ] Fact verification cross-checks claims across sources
- [ ] Staleness detection flags outdated knowledge
- [ ] Citation tracking links claims to sources
- [ ] Contradiction detection in knowledge graph

**Success metric**: Published content has traceable sources and no stale claims.

---

## Layer 7: Output Flywheel (3-4 sessions)
**Goal**: Engagement drives better research.

- [ ] Track which topics/formats get most engagement
- [ ] Research priorities shift toward high-engagement areas
- [ ] A/B test different content formats
- [ ] Community feedback loop (comments/replies inform research)
- [ ] Follower growth tracking

**Success metric**: Measurable improvement in engagement over 2 weeks.

---

## Layer 8: Cross-Domain Synthesis (2-3 sessions)
**Goal**: Insights that span lieutenant domains.

- [ ] War Room debates produce cross-domain analysis
- [ ] Knowledge graph connects entities across domains
- [ ] Synthesis reports identify cross-cutting trends
- [ ] "Unexpected connection" detection between domains
- [ ] Multi-lieutenant research tasks

**Success metric**: At least 3 published insights that combine knowledge from 2+ domains.

---

## Layer 9: Empire Spawning (3-4 sessions)
**Goal**: Replicate Empire for other verticals.

- [ ] Empire template with configurable domains
- [ ] Cross-empire knowledge bridge shares insights
- [ ] Empire registry tracks all running instances
- [ ] One-click deploy for new verticals (crypto, biotech, etc.)
- [ ] Shared infrastructure (LLM routing, budget management)

**Success metric**: Second Empire instance running autonomously on a different vertical.

---

## Layer 10: God Panel (4-5 sessions)
**Goal**: Unified command interface for all Empires.

- [ ] Dashboard showing all Empire health/activity
- [ ] Direct command injection to any lieutenant
- [ ] Budget management across Empires
- [ ] Research priority override
- [ ] Kill switch and rate limiting
- [ ] Real-time activity feed

**Success metric**: Single interface to monitor and control multiple Empires.

---

## Layer 11: Real-World Actions (ongoing)
**Goal**: Empire acts, not just observes.

- [ ] Auto-create GitHub repos for promising research directions
- [ ] Submit to conferences/workshops
- [ ] Engage with researchers on social media
- [ ] Contribute to open-source projects
- [ ] Generate and publish datasets

**Success metric**: Empire produces artifacts beyond text content.

---

## Layer 12: Self-Modification (the long game)
**Goal**: Empire rewrites its own code.

- [ ] Evolution proposals include code changes
- [ ] Automated testing validates proposed changes
- [ ] Staging environment for testing modifications
- [ ] Rollback mechanism for failed changes
- [ ] Empire optimizes its own prompts, tools, and architecture

**Success metric**: Empire successfully deploys a self-authored improvement.

---

## Cost Estimates

| Layer | Estimated LLM Cost | Sessions |
|-------|-------------------|----------|
| 0-1   | $10-20            | 2-4      |
| 2-3   | $20-40            | 4-6      |
| 4-5   | $30-50            | 4-6      |
| 6-7   | $40-60            | 5-7      |
| 8-9   | $30-50            | 5-7      |
| 10    | $30-55            | 4-5      |
| **Total to L10** | **$160-275** | **24-35** |

Layers 11-12 are ongoing with variable costs.
