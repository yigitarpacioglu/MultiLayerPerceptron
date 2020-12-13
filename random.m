function W = random(rows, columns)

W = zeros(columns, 1 + rows);

    epsilon_init = 0.1;
    
    W = rand(columns, 1 + rows) * 2 * epsilon_init - epsilon_init;


end
