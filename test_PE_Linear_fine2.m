clc,clear;


% van Waarde H J. Beyond persistent excitation: Online experiment design 
% for data-driven modeling and control[J]. 
% IEEE Control Systems Letters, 2021, 6: 319-324.


% linear system : 
% x(t+1) = A * x(t) + B * u(t)
% y(t) = C * x(t)

% random A, B, C
n = 100;
m = 8;
p = 15;


disp(['Generate random linear system (A, B) ...'])

C = [eye(p), zeros(p, n-p)];

while 1
    A = random('uniform', -1, 1, [n, n]);
    B = random('uniform', -1, 1, [n, m]);
    % rescale the eigenvalues of A     
    A = A / abs(eigs(A, 1, 'LM'));
    B = B / abs(eigs(A, 1, 'LM'));
    % controllability
    Co = ctrb(A, B);  % size (n, n * m)
    % observability
    Ob = obsv(A, C);   % size (n * p, n)
    if rank(Co) == n && rank(Ob) == n
        break;
    end
end


%% Generate persistently exciting data using online design

Lag = n;
while 1
    if rank(Ob(1 : p * (Lag - 1), :)) == n
        Lag = Lag - 1;
    else
        break;
    end
end

T = Lag + 2;  % T > Lag
L = n+(m+1)*T-1 ;   % L = n+(m+1)*T-1, number of samples

disp(['Prediction length : T = ', num2str(T)])

disp(['Generate control using online design...'])

% U(1 : T) is arbitrarily chosen
U_ = random('uniform', -0.5, 0.5, [m, T]);

X_ = random('uniform', -2, 2, [n, 1]);
for i = 1 : T - 1
    X_ = [X_, A * X_(:, i) + B * U_(:, i)];
end
Y_ = C * X_;

% online design of U(T+1 : L) 
for i = T+1 : L
    Stack = [Hankel(Y_(:, 1 : i-2), p, i-2, T-1); 
             Hankel(U_(:, 1 : i-2), m, i-2, T-1)];
         
    X_ = [X_, A * X_(:, i-1) + B * U_(:, i-1)];
    Y_ = C * X_;
    
    yu = [vec(Y_(:, i-T+1 : i-1)); vec(U_(:, i-T+1 : i-1))];
    
    if rank([Stack, yu]) > rank(Stack)
        % choose u(i) arbitrarily
        U_ = [U_, random('uniform', -0.5, 0.5, [m, 1])];
    else
        Stack_new = [Hankel(Y_(:, 1 : i-2), p, i-2, T-1); 
                     Hankel(U_(:, 1 : i-1), m, i-1, T)];
        null_vec = null(Stack_new');
        if size(null_vec, 1) ~= (p*(T-1) + m*T)
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
            yu_new = [vec(Y_(:, i-T+1 : i-1)); 
                      vec(U_(:, i-T+1 : i-1)); 
                      vec(u_new)];
            if null_v' * yu_new ~= 0
                U_ = [U_, u_new];
                break;
            end
        end
    end
end


disp(['Generate data samples (u, y) ...'])

for i = T : L-1
    X_ = [X_, A * X_(:, i) + B * U_(:, i)];
end
Y_ = C *X_;

disp(['Verification of row rank of Hankel (Y, U)'])

rank([Hankel(Y_, p, L, T); Hankel(U_, m, L, T)]) == n + m * T



% check the Willems' Fundamental Lemma

U_pred = random('uniform', -1, 1, [m, T]);
X_pred = zeros(n, T);
X_pred(:, 1) = random('uniform', -2, 2, [n, 1]);

for i = 1 : T - 1
    X_pred(:, i+1) = A * X_pred(:, i) + B * U_pred(:, i);
end

Y_pred = C * X_pred;

% training data
G = [Hankel(Y_, p, L, T); Hankel(U_, m, L, T)];

% right hand side
A = [vec(Y_pred); vec(U_pred)];


% D * alpha = Y, alpha is a vector of length L-T+1

alpha = lsqminnorm(G, A);


disp(['alpha error: ', num2str(norm(G * alpha - A))])

