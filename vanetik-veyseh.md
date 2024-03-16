# notes on Vanetik et al. and Veyseh et al.
## Vanetik: "Automated Discovery of Mathematical Definitions in Text"
- present new dataset for defext on math (called WFM)
    - manually annotated from Wolfram MathWorld
    - about 20% more non-definitions than definitions
    - more articles than sentences???
    - link: https://github.com/uplink007/FinalProject/tree/master/data/wolfram
- TODO: test **our code** on **their annotations**
- they implemented a bunch of different neural network structures
    - don't think it's feasible for us to test them all
    - link to all: https://github.com/uplink007/FinalProject/tree/master/model
    - best performances on the WFM (two different embedding types) are around 0.78
- in all they "propose a novel representation for text sentences based solely on the dependency information, and its combination with deep neural networks, to handle [definiton extraction]. 
    - is this something? do we want to use only dependencies? I didn't really understand how they got this info

## Veyseh: "A Joint Model for Definition Extraction with Syntactic Connection and Semantic Consistency"
- do sentence classification and "sequence labeling" (i.e. saying what term is being defined) simultaneously
- TODO: test **their code** on **our annotations** (and Vanetik annotations)
    - https://github.com/amirveyseh/definition_extraction
- they aren't domain-specific like we are (and Vanetik et al. claim to be)
    - they do merely OK on sentence classificaiton, but that's comparable to or better than SOTA
    - they do much better on the sequence labeling part
    - they do worst on sentence classification from textbooks