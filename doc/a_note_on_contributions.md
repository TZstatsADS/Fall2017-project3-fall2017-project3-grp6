### A Note on Contributions

Whenever we have team projects, there are always concerns on unequal contributions from members of a project team. In the ideal world, we are all here to put in our best efforts and learn together. Even in that ideal world, we have different skill sets and preparations, and we will still contribute differently to a project. 

Therefore, you are required to post a *contribution statement* in the root README.md of your GitHub repo. Please beware that your GitHub repo will become public and remain public after the due date of the projects. 

Post your title, team members, project abstract and a contribution statement in the README.md file.  This is a common practice for research scientific journals. 

Below is an example. If no contribution statement is provided, we will insert a default statement that goes "**All team members contributed equally in all stages of this project. All team members approve our work presented in this GitHub repository including this contributions statement**. "

---
Sample project README statement.

Project 3

Team members: Sijian Xuan, Xinyao Guo, Siyi Wang, Pinren Chen, Xiaoyu Zhou

Summary: In this project, we used several feature extraction such as LBP feature extraction, HoG feature extraction methods and classification models from machine learning to recogonize whether there is a dog, chicken or blueberry muffin in the image. We have a baseline model which is GBM + SIFT and we are trying to develop a way that improve the baseline model most. We also tried RGB for feature extraction and CNN for both feature extraction and classification. However, there are grayscale images that RGB could not deal with; and CNN takes a long time to train the model. We finally use SVM + LBP and xgboost + HoG as our winners. They achieve a accracy rate of about 80% and takes a short time to train (less than 1 minute).

[Contribution Statement]
Sijian Xuan: as the group presenter, is working on the whole organization of the study. He does the research about LBP feature extraction method and choose BPNN, SVM, Random Forest, Logistic regression as well as writing the relevant code with help of Siyi Wang and Xinyao Guo. He collects everyone's code and write them in main.Rmd and write the ppt file.

Xinyao Guo: tried HoG and RGB to do feature extraction and finally chose HoG. Applied logistic regression(multinomial), random forest, KNN, xgboost as candidate classification models. Tune the models by grid searching method. Discard KNN as the accuracy is only around 60%. Applied cross validation to prevent the overfitting issue.

Xiaoyu Zhou: was responsible for CNN model analysis. Conducted CNN in R for both feature extraction and classification, decreased the baseline error rate successfully but discarded R-version due to the time spend. Collaborated with Pinren to write code and debug in Python for CNN feature extraction and classification, and achieved the accuracy rate >90%. Accelerated the Python running time.
