function bounDary(w, X, y)
% This function plots the data points X and y into a new figure with

% Plot Data
plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3
    
    
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    plot_y = (-1./w(3)).*(w(2).*plot_x + w(1));
    plot(plot_x, plot_y)
    axis([-4,4,-4,4]);
   
hold off

end