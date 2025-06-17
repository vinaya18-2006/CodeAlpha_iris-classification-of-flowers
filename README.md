
🌸 C++ Iris Flower Classifier using k-Nearest Neighbors (k-NN)

🔍 A Machine Learning Project from Scratch in C++

I recently built a mini machine learning classifier in C++ to recognize the species of Iris flowers using the k-NN algorithm — without any external libraries or datasets. It’s lightweight, fast, and works even in online compilers like OnlineGDB.
💡 Project Overview

The Iris dataset contains measurements of three flower species:

🌺 Iris-setosa

🌿 Iris-versicolor

🌸 Iris-virginica

Each flower is described using:

Sepal Length

Sepal Width

Petal Length

Petal Width

The goal: classify an unseen flower based on these measurements using the k-nearest neighbors algorithm (k=3).
🔧 Functional Breakdown

🧱 Block / Function	🔍 Purpose

struct Iris	Holds the 4 features + species label of a sample
euclideanDistance()	Calculates similarity between two flower feature vectors
classifyKNN()	Predicts the flower species using k-NN logic
loadEmbeddedDataset()	Parses and loads sample Iris data embedded in code
splitDataset()	Randomly splits data into training and test sets
main()	Orchestrates the process: training → prediction → evaluation
⚙️ Technical Highlights

✅ Implemented k-NN from scratch

✅ No external file I/O — data embedded directly into the C++ source

✅ Works on OnlineGDB, Replit, or any standard C++ compiler

✅ Achieved 100% accuracy on sample test set with k = 3

🧠 What I Learned

How ML algorithms like k-NN work under the hood

Data parsing, normalization, and classification logic in C++

Performing data science tasks without Python libraries like NumPy or Pandas
#Cplusplus #MachineLearning #kNN #IrisDataset #CppProjects #OnlineGDB #ArtificialIntelligence #DataScience #FromScratch #OpenSourceLearning #MLinCpp #Programming
