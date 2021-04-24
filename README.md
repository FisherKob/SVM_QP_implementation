# SVM_QP_implementation

In this module, I implemented SVM using convex optimization concept called Quadtratic programming. SVM is a primal/dual problem. Using dual variables, we can define dual function which can be shown to acheive KKT optimality condition.
Using that formulation, we can get B^* and B_{0}. With that result, we can fit set up SVM as a QP problem, simulate the model, fit the training sets, and predict on testing set.
