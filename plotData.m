function plotData(X, td)
%   Plots the data points X and y into a new figure 

% Create New Figure
figure; hold on;

   class1=find(td(:,1)==0 & td(:,2)==0);
   class2=find(td(:,1)==0 & td(:,2)==1);
   class3=find(td(:,1)==1 & td(:,2)==0);

   
    
    plot(X(class1, 1), X(class1, 2), 'rx','LineWidth', 2, 'MarkerSize', 7);
    plot(X(class2, 1), X(class2, 2), 'g+','LineWidth', 2, 'MarkerSize', 7)
    plot(X(class3, 1), X(class3, 2), 'bo','LineWidth', 2, 'MarkerSize', 7)
    
    
    
    
    
hold off;

end