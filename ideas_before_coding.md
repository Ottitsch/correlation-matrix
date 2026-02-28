# Time Limit 2 Hours

### My initial vague Ideas before doing any research:
1. simple semantic embedding
2. Ai generate additional data for a rubric and compare that either via Graphs or Semantics
3. Genetic AI approach


## What I actually will decide to do after my research:

Methodology:
1. Understanding Task and Researching 
2. ~~Manually Created Small Eval~~ | stole one ground truth sample from https://demo.romagnolo.ai/documentation/screenshots/marked/Romagnolo_Screenshot_marked_de_SearchField1.png
and decided to select based on that

3. Tell Claude to generate one AI slop approach
4. Check The Leaderboard of https://talentclef.github.io/talentclef/docs/talentclef-2025/task-summary for TASK A Results (en-de)
5. Implement ALEXU-NLP approach
6. Implement pjmathematician approach
7. Create Ensemble approach
8. Judge each approach by checking which got closest to romagnolo.ai sample
9. Submit closest approach and tidy up repo
10. Outlook and Discussion


[Gonna use TechWolf as a Backup paper in case, ALEXU-NLP or pjmathematician don't work]

------------------------

What would I do if I didn't have the Ground Truth Sample from romagnolo.ai?  
-> Manually Judge Some Jobs related to software engineering

Why manual?  
180 work fields is a small enough data sample for a manual Eval or domain Experts to be relevant without substantial costs.  
(I consider myself a domain expert in swe).

IF I had much more time I would spend more time on expanding the eval more either   
by interviewing domain Experts and/or judging more than just 1 job title  
OR spending more time searching for existing evals online and seeing how I can adapt those to my specific problem

Other Approaches I'd be interested in experimenting with additional time:  
Playing around with different Model Sizes, either by using AZURE AI or Google Vertex AI, that way the Data remains GDPR compliant, in case we find out bigger Models work better and our client insists on using GDPR compliant Models.  
AND exploring if smaller finetuned models can beat everything  
(small data beats big data)  
🙏💖

----------------------
### Citations:  
@inproceedings{gasco2025overview,
  title={{Overview of the TalentCLEF 2025: Skill and Job Title Intelligence for Human Capital Management}},
  author={Gasco, Luis and Fabregat, Hermenegildo and Garc\'{\i}a-Sardi{\~n}a, Laura and Estrella, Paula and Deniz, Daniel and Rodrigo, \'{A}lvaro and Zbib, Rabih},
  booktitle={{Experimental IR Meets Multilinguality, Multimodality, and Interaction}},
  publisher={{Springer Nature Switzerland}},
  address={{Cham}},
  pages={{464--485}},
  isbn={{978-3-032-04354-2}},
  year={2025},
}
    
@inproceedings{alexunlp2025,
  title={{AlexU-NLP at TalentCLEF 2025: Curriculum-Driven Hybrid Retrieval for Multilingual Job Title Matching}},
  author={Rana Barakat, Omar Mokhtar, Marwan Torki and Nagwa Elmakky},
  booktitle={{CLEF (Working Notes)}},
  year={2025}
}

@inproceedings{pjmathematician2025,
  title={{pjmathematician at TalentCLEF 2025: Enhancing Job Title and Skill Matching with GISTEmbed and LLM-Augmented Data}},
  author={Poojan Vachharajani},
  booktitle={{CLEF (Working Notes)}},
  year={2025}
}

@inproceedings{techwolf2025,
  title={{Multilingual JobBERT for Cross-Lingual Job Title Matching}},
  author={Jens-Joris Decorte, Matthias De Lange and Jeroen Van Hautte},
  booktitle={{CLEF (Working Notes)}},
  year={2025}
}
