% #######################################################################
%
% YILDIZ TECHNICAL UNIVERSITY
% Faculty of Electrical & Electronics Engineering
% Control and Automation Engineering Department
% 
% -----------------------------------------------------------------------
%
% Yiğit ARPACIOĞLU
% 18567037
% yigit.arpacioglu@gmail.com
% 
% -----------------------------------------------------------------------
%
% This script is written to train and test neural network with 
% class3_tr and class3_test data sets.
%
% -----------------------------------------------------------------------
% #######################################################################

clc; clear all; close all;

%% Part 1: Reading and Visualizing Data

traindata = load('class3_tr.txt'); 
% data is read from txt file and assigned

X=traindata(:,1:2); td = traindata(:,3:4);                                  
%Data randomized

% Visualizing is provided with another function called 'plotData'
plotData(X,td), 
title('Training Data Distribution')
xlabel('Feature 1')
ylabel('Feature 2')
legend('Class 1', 'Class 2', 'Class 3')

% For using vectorized operations, there is a need to add a column vector 
% that consists from 'ones'. (That colum will be multiplied with bias) x0=1

%% Part 2: Initializing Neural Network Parameters

[m,n] = size(X);

input_layer_size  = 2;   % There are 2 features
hidden_layer_size = 2;   % There is no need for more hidden units as in that simple case 
num_labels = 3;          % 3 labels, that we desired to seperate into 3 classes
X = [ones(m,1) X]; 
% Getting user input for learning rate and iteration

max_iter=input('Please enter a suitable iteration number for learning\n'); 
alpha=input('Please enter learning rate\n');
J_history=zeros(max_iter,1);

% Random weight initialization
in_w1 = random(input_layer_size, hidden_layer_size);
in_w2 = random(hidden_layer_size, num_labels);


% Load the initial weights into variables w1 and w2
w1=in_w1; w2=in_w2;


iter = 1; % iteration number initialized

cl1_ind=find(td(:,1)==0 & td(:,2)==0);
cl2_ind=find(td(:,1)==0 & td(:,2)==1);
cl3_ind=find(td(:,1)==1 & td(:,2)==0);

td(cl1_ind,:)=1;
td(cl2_ind,:)=2;
td(cl3_ind,:)=3;

temp = eye(num_labels);
td_new = temp(td,:);
td_new = td_new(1:330,:);

%% Part 3 Neural Network

while iter<max_iter
% Forward Propagation

z2 = X*w1'; 
a2 = sigmoid(z2); %

a2 = [ones(m,1) a2];
z3 = a2*w2';
od = sigmoid(z3); 

%logistic regression
J = 1/m * sum(sum(-td_new .* log(od) - (1 - td_new) .* log(1 - od)));
J_history(iter)=J; 

% Backward Propagation

delta_3 = od - td_new;
delta_2 = ((delta_3 * w2(:,2:end)) .* sig_grad(z2));

w1_grad = 1/m * delta_2' * X;
w2_grad = 1/m * delta_3' * a2;

%Weight Update
grad=[ w1_grad(:);w2_grad(:)];

w=[w1(:);w2(:)];

w=w-alpha*grad;

w1=reshape(w(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
w2=reshape(w(1 + (hidden_layer_size * (input_layer_size + 1)):end),num_labels, (hidden_layer_size + 1));

iter=iter+1;

end

pred = pred(w1, w2, X);

fprintf('\nTraining Set Accuracy: %f\n', (mean(double(pred == td(:,1)) * 100)+ mean(double(pred == td(:,2)) * 100))/2);
acc1=(mean(double(pred == td(:,1)) * 100)+ mean(double(pred == td(:,2)) * 100))/2;
acc1=round(acc1,2);
bounDary(w, X, td_new)
title(['Training Set Accuracy: %', num2str(acc1)])

%% Part 4: Test Data

testdata = load('class3_test.txt');          % data is read from txt file and assigned to
X_t=testdata(:,1:2); td_t = testdata(:,3:4);   % assigned to seperate matrices.

plotData(X_t,td_t), 
title('Test Data Distribution')
xlabel('Feature 1')
ylabel('Feature 2')
legend('Class 1', 'Class 2', 'Class 3')


[m_t, n_t] = size(X_t);

X_t=[ones(120,1) X_t];
z2 = X_t * w1';
a2 = [ones(size(sigmoid(z2), 1), 1) sigmoid(z2)];

z3 = a2 * w2';
od_t = sigmoid(z3);

[x, ix] = max(od_t, [], 2);
p = ix;

cl1_ind_t=find(td_t(:,1)==0 & td_t(:,2)==0);
cl2_ind_t=find(td_t(:,1)==0 & td_t(:,2)==1);
cl3_ind_t=find(td_t(:,1)==1 & td_t(:,2)==0);

td_t(cl1_ind_t,:)=1;
td_t(cl2_ind_t,:)=2;
td_t(cl3_ind_t,:)=3;

temp_t = eye(num_labels);
td_new_t = temp_t(td_t,:);
td_new_t = td_new_t(1:120,:);



fprintf('\nTest Set Accuracy: %f\n', (mean(double(p == td_t(:,1)) * 100)+ mean(double(p == td_t(:,2)) * 100))/2);
acc=(mean(double(p == td_t(:,1)) * 100)+ mean(double(p == td_t(:,2)) * 100))/2;
acc=round(acc,2);
bounDary(w, X_t, td_new_t)
title(['Test Set Accuracy: % ', num2str(acc)])

   
