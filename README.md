# correlationCAUSATION
Wanted to make a AI that could web scrape and register any possible, good trades and said "why not make fun of anyone who says one thing is caused by something else rather than the two just being correlated?". While I hope this will also work as a stock tracker ML model, I'm gonna make it possible to be used for the second purpose for fun.

Goal: Build an end-to-end system to identify causal vs. correlational relationships in real-world news and economic indicator data. Maybe use it for other 

## 📌 Project Status
**Current Stage:** *(e.g., Planning / Early Development / MVP / Production)*  
**What’s Done:**  
- Basic File setup for project
- env and config file skeletons 
**Next Steps:**  
- Start cataloging training data
- Develop a simple neural network for preliminary training

project-root/
│
├── README.md
├── config.yaml
├── env.yml
├── docker-compose.yml
│
├── db/
│ ├── schema.sql
│ └── db_utils.py
│
├── pipelines/
│ ├── ingest_news.py
│ └── ingest_indicators.py
│
├── ai_magic/
│ └── train_infer.py
│
└── cluster/
└── sbatch/
