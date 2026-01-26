from google.genai import types
from src.utils.gemini_schema import EvLevelClassificationCivic


def system_prompt():
    return "You are a researcher that has experience on assessing the evidence level of publications."


def user_prompt_one_shot(title, abstract):
    return f"""Analyse the title and abstract from a biomedical publication below within brackets:
    Title: "{title}"
    Abstract: "{abstract}"
        
    Your task: assign this publication to a CIViC evidence level (A, B, C, D, or E) based only on the type of study and strength of evidence.
    Evidence Level Definitions:
    Level A — Validated Association: Proven/consensus association in human medicine. Examples: Phase III clinical trials, practice guidelines, FDA-approved diagnostics. Practice-changing evidence.
    Level B — Clinical Evidence: Clinical trials (Phase I–III) or primary patient data with >5 patients. Has statistical results but less definitive than Level A.
    Level C — Case Study: Case reports or very small patient series (<5 patients). Often single-patient reports in clinical journals.
    Level D — Preclinical Evidence: In vivo or in vitro models (e.g., mice, cell lines). Not direct human evidence.
    Level E — Inferential Association: Indirect/inferred evidence (e.g., computational prediction, hypothesized mechanism, no direct measurement).
    Unsure — Unable to assign any of the above levels.
    
    For your response:
    1. Assign the evidence level.
    2. Provide a brief explanation of your reasoning.
    3. Indicate your confidence in the classification (e.g., high, medium, low).
    
    Respond **only** with JSON."""


def user_prompt_few_shot(title, abstract, train_dataset):
    one_shot = user_prompt_one_shot(title, abstract)

    # Few-shot examples selected from training dataset
    # Here we provide 5 examples, one for each evidence level
    # Model outputs were obtained from prior runs
    few_shot = """  
    ### Example 1
    **Title:** "Osimertinib: First Global Approval."
    **Abstract:** "Osimertinib (Tagrisso(™), AZD9291) is an oral, third-generation epidermal growth factor receptor tyrosine kinase inhibitor (EGFR TKI) that is being developed by AstraZeneca for the treatment of advanced non-small cell lung cancer (NSCLC). Osimertinib has been designed to target the EGFR T790M mutation that is often present in NSCLC patients with acquired EGFR TKI resistance, while sparing wild-type EGFR. In November 2015, the tablet formulation of osimertinib was granted accelerated approval in the USA for the treatment of patients with metastatic EGFR T790M mutation-positive NSCLC (as detected by an FDA-approved test) who have progressed on or after EGFR TKI therapy. Osimertinib has also been granted accelerated assessment status for this indication in the EU, and is in phase III development for first- and second-line and adjuvant treatment of advanced EGFR mutation-positive NSCLC in several countries. Phase I trials in patients with advanced solid tumours are also being conducted. This article summarizes the milestones in the development of osimertinib leading to this first approval for NSCLC."
    **Model Output:**
    ```json
    {{
    "evidence_level": "A",
    "reasoning": "The abstract describes FDA approval and regulatory validation for osimertinib in NSCLC, representing a proven clinical association in human medicine.",
    "confidence": "high"
    }}
    
    ### Example 2
    **Title:** "DNMT3A mutations in acute myeloid leukemia."
    **Abstract:** "The genetic alterations responsible for an adverse outcome in most patients with acute myeloid leukemia (AML) are unknown.Using massively parallel DNA sequencing, we identified a somatic mutation in DNMT3A, encoding a DNA methyltransferase, in the genome of cells from a patient with AML with a normal karyotype. We sequenced the exons of DNMT3A in 280 additional patients with de novo AML to define recurring mutations.A total of 62 of 281 patients (22.1%) had mutations in DNMT3A that were predicted to affect translation. We identified 18 different missense mutations, the most common of which was predicted to affect amino acid R882 (in 37 patients). We also identified six frameshift, six nonsense, and three splice-site mutations and a 1.5-Mbp deletion encompassing DNMT3A. These mutations were highly enriched in the group of patients with an intermediate-risk cytogenetic profile (56 of 166 patients, or 33.7%) but were absent in all 79 patients with a favorable-risk cytogenetic profile (P<0.001 for both comparisons). The median overall survival among patients with DNMT3A mutations was significantly shorter than that among patients without such mutations (12.3 months vs. 41.1 months, P<0.001). DNMT3A mutations were associated with adverse outcomes among patients with an intermediate-risk cytogenetic profile or FLT3 mutations, regardless of age, and were independently associated with a poor outcome in Cox proportional-hazards analysis.DNMT3A mutations are highly recurrent in patients with de novo AML with an intermediate-risk cytogenetic profile and are independently associated with a poor outcome. (Funded by the National Institutes of Health and others."
    **Model Output:**
    ```json
    {{
    "evidence_level": "B",
    "reasoning": "This study analyzed over 280 patient samples, providing statistically significant associations between DNMT3A mutations and clinical outcomes, consistent with strong but not definitive clinical evidence.",
    "confidence": "high"
    }}
    
    ### Example 3
    **Title:** "Response and acquired resistance to everolimus in anaplastic thyroid cancer."
    **Abstract:** "Everolimus, an inhibitor of the mammalian target of rapamycin (mTOR), is effective in treating tumors harboring alterations in the mTOR pathway. Mechanisms of resistance to everolimus remain undefined. Resistance developed in a patient with metastatic anaplastic thyroid carcinoma after an extraordinary 18-month response. Whole-exome sequencing of pretreatment and drug-resistant tumors revealed a nonsense mutation in TSC2, a negative regulator of mTOR, suggesting a mechanism for exquisite sensitivity to everolimus. The resistant tumor also harbored a mutation in MTOR that confers resistance to allosteric mTOR inhibition. The mutation remains sensitive to mTOR kinase inhibitors."
    **Model Output:**
    ```json
    {{
    "evidence_level": "C",
    "reasoning": "The abstract reports findings from a single patient case describing treatment response and resistance mechanisms, characteristic of a case study.",
    "confidence": "high"
    }}
    
    ### Example 4
    **Title:** "Crizotinib, a c-Met inhibitor, prevents metastasis in a metastatic uveal melanoma model."
    **Abstract:** "Uveal melanoma is the most common primary intraocular malignant tumor in adults and half of the primary tumors will develop fatal metastatic disease to the liver and the lung. Crizotinib, an inhibitor of c-Met, anaplastic lymphoma kinase (ALK), and ROS1, inhibited the phosphorylation of the c-Met receptor but not of ALK or ROS1 in uveal melanoma cells and tumor tissue. Consequently, migration of uveal melanoma cells was suppressed in vitro at a concentration associated with the specific inhibition of c-Met phosphorylation. This effect on cell migration could be recapitulated with siRNA specific to c-Met but not to ALK or ROS1. Therefore, we developed a uveal melanoma metastatic mouse model with EGFP-luciferase-labeled uveal melanoma cells transplanted by retro-orbital injections to test the effect of crizotinib on metastasis. In this model, there was development of melanoma within the eye and also metastases to the liver and lung at 7 weeks after the initial transplantation. When mice were treated with crizotinib starting 1 week after the transplantation, we observed a significant reduction in the development of metastases as compared with untreated control sets. These results indicate that the inhibition of c-Met activity alone may be sufficient to strongly inhibit metastasis of uveal melanoma from forming, suggesting crizotinib as a potential adjuvant therapy for patients with primary uveal melanoma who are at high risk for the development of metastatic disease."
    **Model Output:**
    ```json
    {{
    "evidence_level": "D",
    "reasoning": "The study used cell line assays and mouse xenograft models to test drug effects, representing preclinical in vitro and in vivo evidence without human subjects.",
    "confidence": "high"
    }}
    
    ### Example 5
    **Title:** "Targeting Akt3 signaling in triple-negative breast cancer."
    **Abstract:** "Triple-negative breast cancer (TNBC) is currently the only major breast tumor subtype without effective targeted therapy and, as a consequence, in general has a poor outcome. To identify new therapeutic targets in TNBC, we performed a short hairpin RNA (shRNA) screen for protein kinases commonly amplified and overexpressed in breast cancer. Using this approach, we identified AKT3 as a gene preferentially required for the growth of TNBCs. Downregulation of Akt3 significantly inhibits the growth of TNBC lines in three-dimensional (3D) spheroid cultures and in mouse xenograft models, whereas loss of Akt1 or Akt2 have more modest effects. Akt3 silencing markedly upregulates the p27 cell-cycle inhibitor and this is critical for the ability of Akt3 to inhibit spheroid growth. In contrast with Akt1, Akt3 silencing results in only a minor enhancement of migration and does not promote invasion. Depletion of Akt3 in TNBC sensitizes cells to the pan-Akt inhibitor GSK690693. These results imply that Akt3 has a specific function in TNBCs; thus, its therapeutic targeting may provide a new treatment option for this tumor subtype."
    **Model Output:**
    ```json
    {{
    "evidence_level": "E",
    "reasoning": "The abstract describes inferred therapeutic hypotheses derived from molecular and functional screening, suggesting potential but unvalidated mechanisms rather than direct human or clinical evidence.",
    "confidence": "high"
    }}
    """
    return one_shot + "\n" + few_shot


def gpt_prompt(
    title, abstract, client, model, user_prompt_type, json_schema, temperature
):
    """GPT-specific prompt wrapper."""

    # Generate content with the adapted parameters
    if user_prompt_type == "one_shot":
        user_prompt = user_prompt_one_shot(title, abstract)
    elif user_prompt_type == "few_shot":
        user_prompt = user_prompt_few_shot(title, abstract, None)
    else:
        raise ValueError("Invalid user_prompt_type. Choose 'one_shot' or 'few_shot'.")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_schema", "json_schema": json_schema},
    )

    return response.choices[0].message.content


def gemini_prompt(title, abstract, client, model, user_prompt_type, temperature):
    """Gemini-specific prompt wrapper."""

    # Generate content with the adapted parameters
    if user_prompt_type == "one_shot":
        contents = user_prompt_one_shot(title, abstract)
    elif user_prompt_type == "few_shot":
        contents = user_prompt_few_shot(title, abstract, None)
    else:
        raise ValueError("Invalid user_prompt_type. Choose 'one_shot' or 'few_shot'.")

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt(),
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=list[EvLevelClassificationCivic],
        ),
    )

    return response.text
