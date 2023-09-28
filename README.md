# Blog_interactions

Measuring feature interactions with SHAP and ALE, and exploring why the results are conflicting.  Code for a blog to be submitted to [Towards Data Science](https://towardsdatascience.com).

To run, do the following:

#  Download the Lending Club Loans data from https://www.kaggle.com/datasets/wordsforthewise/lending-club?resource=download
#  Modify 00_setup.py
  # input_path should point to the location of the file "accepted_2007_to_2018Q4.csv" on your system
  # temp_path should point to a writeable directory on your system (intermediate data files, plots, etc. will be placed here)
# Run scripts in numeric order

The notebooks compare SHAP, ALE, and Friedman's H measures of interaction strength.  They especially focus on the interaction between interest rate and term, because SHAP and ALE disagree dramatically on the direction of that interaction.  

I find that ALE signals reflect customers with several risk factors, where term enhances risk.  The model's response to a rare combination of features leads to this behavior.  Because the feature combination is rare, and SHAP averages many coalitions, this rare combination does not affect SHAP values much.  Instead, SHAP responsds to more systematic differences where high interest rates capture most of the default risk.

This blog illustrates some pitfalls and considerations when using model explainability techniques, and emphasizes that understanding test methodology is crucial for proper use of explainability techniques.