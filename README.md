# cancerprediction
Understanding Cancer with Metastasis Data

Cancer: the #2 cause of death in the United States among adults. Over a half-million Americans die from cancer every year, and around 40% of people develop cancer at some point in their lives (cancer.gov). One sentence that no cancer patient wants to hear is: “Your cancer is metastatic.” Metastasis means the cancer has spread from its origin to other parts of the body, via the bloodstream or lymph nodes. It is vital for patients and medical professionals to know whether one’s cancer is metastatic, so that they may rapidly deploy the necessary treatments to mitigate the spread.

But while it is crucial for treatment purposes to test for metastasis, does these test results give any clearer insights into the outcomes of cancer patient? Studies have yet to give conclusive answers one way or the other. This investigation aims to determine whether the outcome of cancer cases is more predictable once the cancer is classified with or without distant metastasis. The outcome of this research will help cancer patients understand the clarity of their future once they undergo metastasis testing. It will also assist medical providers in making more accurate decisions on when to rule a cancer case as terminally ill or curable.

Refer to the file CancerInvestigationReport.pdf for the project explanation and analysis.

The workflow of the code analysis is:

1. Download the files colorectal.csv, prostate.csv and esophageal.csv.
2. Load these three files into the working file directory of R, and run the code DataCleaning.R in an R IDE, which will clean the three dataframes, merge them, and automatically write the merged dataframe called cancerdatamerged.csv to the working file directory.
3. Run the file ResolvingNAs.R, which will impute missing values of the cancerdatamerged.csv dataframe and automatically write the new dataframe called finalcancerdata.csv to the working file directory.
4. Run the file PatientPredictions.py in a Python IDE, which fits the machine learning models, builds the ensemble classifer, creates the visualizations and computes the results of the classifiers.
