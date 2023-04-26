clc,clear;


% van Waarde H J. Beyond persistent excitation: Online experiment design 
% for data-driven modeling and control[J]. 
% IEEE Control Systems Letters, 2021, 6: 319-324.


% linear system : x(t+1) = A * x(t) + B * u(t)

% random A and B
n = 100;
m = 8;


disp(['Generate random linear system (A, B) ...'])

while 1
    A = random('uniform', -1, 1, [n, n]);
    B = random('uniform', -1, 1, [n, m]);
    % rescale the eigenvalues of A     
    A = A / abs(eigs(A, 1, 'LM'));
    B = B / abs(eigs(A, 1, 'LM'));
    % controllability
    Co = ctrb(A, B);  % size (n, n x m)
    if rank(Co) == n
        break;
    end
end


%% Generate persistently exciting data using online design

T = 10;
L = n+(m+1)*T-1 ;   % L = n+(m+1)*T-1, number of samples

disp(['Prediction length : T = ', num2str(T)])

disp(['Generate control using online design...'])

% U(1 : T) is arbitrarily chosen
U_ = random('uniform', -0.5, 0.5, [m, T]);

X_ = random('uniform', -2, 2, [n, 1]);
for i = 1 : T - 1
    X_ = [X_, A * X_(:, i) + B * U_(:, i)];
end

% online design of U(T+1 : L) 
for i = T+1 : L
    Stack = [Hankel(X_(:, 1 : i-T), n, i-T, 1); 
             Hankel(U_(:, 1 : i-2), m, i-2, T-1)];
         
    X_ = [X_, A * X_(:, i-1) + B * U_(:, i-1)];
    
    xu = [X_(:, i-T+1); vec(U_(:, i-T+1 : i-1))];
    
    if rank([Stack, xu]) > rank(Stack)
        % choose u(i) arbitrarily
        U_ = [U_, random('uniform', -0.5, 0.5, [m, 1])];
    else
        Stack_new = [Hankel(X_(:, 1 : i-T), n, i-T, 1); 
                     Hankel(U_(:, 1 : i-1), m, i-1, T)];
        null_vec = null(Stack_new');
        if size(null_vec, 1) ~= (n + m*T)
            error('wrong dimension');
        end
        for j = 1 : size(null_vec, 2)
            % nonzero eta_1
            if prod(null_vec(end-m : end, j)) ~= 0
                null_v = null_vec(:, j);
                break;
            end
        end
        % new control
        while 1
            u_new = random('uniform', -1, 1, [m, 1]);
            xu_new = [X_(:, i-T+1); vec(U_(:, i-T+1 : i-1)); vec(u_new)];
            if null_v' * xu_new ~= 0
                U_ = [U_, u_new];
                break;
            end
        end
    end
end


disp(['Generate data samples (u, x) ...'])

for i = T : L-1
    X_ = [X_, A * X_(:, i) + B * U_(:, i)];
end

disp(['Verification of row rank of Hankel (X, U)'])

rank([Hankel(X_, n, L-T+1, 1); Hankel(U_, m, L, T)]) == n + m * T



% check the Willems' Fundamental Lemma

U_pred = random('uniform', -1, 1, [m, T]);
X_pred = zeros(n, T);
X_pred(:, 1) = random('uniform', -2, 2, [n, 1]);

for i = 1 : T - 1
    X_pred(:, i+1) = A * X_pred(:, i) + B * U_pred(:, i);
end

% training data
D = [Hankel(X_, n, L, T); Hankel(U_, m, L, T)];

% right hand side
Y = [vec(X_pred); vec(U_pred)];


% D * alpha = Y, alpha is a vector of length L-T+1

alpha = lsqminnorm(D, Y);


disp(['alpha error: ', num2str(norm(D * alpha - Y))])

