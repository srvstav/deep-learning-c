#include <stdio.h>
#include <math.h>

#define INPUTS 2
#define HIDDEN1 4 
#define HIDDEN2 4 
#define OUTPUTS 1
#define SAMPLES 4

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative_from_output(double y)
{
    return y * (1.0 -y);
}

int main(void)
{
    double X[SAMPLES][INPUTS] = 
    {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    
    double Y[SAMPLES] = {
        0, 1, 1, 0
    };


    double W1[INPUTS][HIDDEN1] = {
    {0.5, -0.3, 0.2, 0.7},
    {-0.4, 0.8 , -0.6, 0.1}
    };

    double b1[HIDDEN1] = {0, 0, 0, 0};

    double W2[HIDDEN1][HIDDEN2] = {
        {0.3, -0.5, 0.2, 0.4},
        {-0.7, 0.1, 0.6, -0.2},
        {0.5, 0.8, -0.3, 0.1},
        {-0.4, 0.2, 0.7, -0.6}
    };

    double b2[HIDDEN2] = {0, 0, 0, 0};

    double W3[HIDDEN2][OUTPUTS] = {
        {0.4},
        {-0.5},
        {0.3},
        {0.7}
    };

    double b3[OUTPUTS] = {0};

    double learning_rate = 1.0;
    
    for(int epoch = 0; epoch < 20000; epoch++)
    {
        double total_loss = 0.0;

        for(int sample = 0; sample < SAMPLES; sample++)
        {
            double z1[HIDDEN1];
            double a1[HIDDEN1];

            for(int j = 0; j< HIDDEN1; j++)
            {
                z1[j] = b1[j];

                for(int i = 0; i < INPUTS; i++)
                {
                    z1[j] += X[sample][i] * W1[i][j];
                }
                a1[j] = sigmoid(z1[j]);

            }
            double z2[HIDDEN2];
            double a2[HIDDEN2];

            for( int j = 0; j < HIDDEN2; j++)
            {
                z2[j] = b2[j];

                for( int i = 0; i < HIDDEN1; i++)
                {
                    z2[j] += a1[i] * W2[i][j];
                }

                a2[j] = sigmoid(z2[j]);

            }
            double z3 = b3[0];

            for(int i = 0; i< HIDDEN2; i++)
            {
                z3 += a2[i] * W3[i][0];
            }

            double output = sigmoid(z3);

            double error = output - Y[sample];
            
            total_loss += error * error;

            double dz3 = 
                error * sigmoid_derivative_from_output(output);

            double dW3[HIDDEN2];
            
            for(int i = 0; i < HIDDEN2; i++)
            {
                dW3[i] = a2[i] * dz3;
            }

            double db3 = dz3;

            double da2[HIDDEN2];
            double dz2[HIDDEN2];

            for(int i = 0; i < HIDDEN2; i++)
            {
                da2[i] = W3[i][0] * dz3;

                dz2[i] = da2[i] * sigmoid_derivative_from_output(a2[i]);
            }

            double dW2[HIDDEN1][HIDDEN2];

            for( int i = 0; i < HIDDEN1; i++)
            {
                for(int j = 0; j < HIDDEN2; j++)
                {
                    dW2[i][j] = a1[i] * dz2[j];
                }
            }

            double db2[HIDDEN2];

            for( int j = 0; j < HIDDEN2; j++)
            {
                db2[j] = dz2[j];
            }

            double da1[HIDDEN1];
            double dz1[HIDDEN1];

            for(int i = 0; i < HIDDEN1; i++)
            {
                da1[i] = 0.0;

                for(int j = 0; j < HIDDEN2; j++)
                {
                    da1[i] += W2[i][j] * dz2[j];
                }

                dz1[i] = da1[i] * sigmoid_derivative_from_output(a1[i]);
            }

            double dW1[INPUTS][HIDDEN1];

            for( int i = 0; i < INPUTS; i++)
            {
                for( int j = 0; j < HIDDEN1; j++)
                {
                    dW1[i][j] = X[sample][i] * dz1[j];
                }
            }

            double db1[HIDDEN1];

            for(int j = 0; j < HIDDEN1; j++)
            {
                db1[j] = dz1[j];
            }

            for( int i = 0; i < INPUTS; i++)
            {
                for( int j = 0; j < HIDDEN1; j++)
                {
                    W1[i][j] -= learning_rate * dW1[i][j];
                }
            }

            for(int j = 0; j < HIDDEN1; j++)
            {
                b1[j] -= learning_rate * db1[j];
            }

            for( int i = 0; i < HIDDEN1; i++)
            {
                for( int j = 0; j < HIDDEN2; j++)
                {
                    W2[i][j] -= learning_rate * dW2[i][j];
                }
            }

            for( int j = 0; j < HIDDEN2; j++)
            {
                b2[j] -= learning_rate * db2[j];
            }

            for( int i = 0 ; i < HIDDEN2; i++)
            {
                W3[i][0] -= learning_rate * dW3[i];
            }

            b3[0] -= learning_rate * db3;
        }

        if(epoch % 1000 == 0)
        {
            printf("Epoch %d | Loss = %.8f\n",
            epoch, total_loss);

        }

    }

    printf("\nFINAL PREDICTIONS: \n");

    for( int sample = 0; sample < SAMPLES; sample++)
    {
        double a1[HIDDEN1];
        
            for(int j = 0; j < HIDDEN1; j++)
            {
            double z = b1[j];

            for(int i = 0; i < INPUTS; i++)
            {
                z += X[sample][i] * W1[i][j];
            }

            a1[j] = sigmoid(z);
            }

        double a2[HIDDEN2];

            for(int j = 0; j < HIDDEN2; j++)
    {
        double z = b2[j];

        for(int i = 0; i < HIDDEN1; i++)
        {
            z += a1[i] * W2[i][j];
        }

        a2[j] = sigmoid(z);
    }
    

    double z3 = b3[0];

    for(int i = 0; i < HIDDEN2; i++)
    {
        z3 += a2[i] * W3[i][0];
    }

    double output = sigmoid(z3);

    printf("%d XOR %d = %.6f\n",
        (int)X[sample][0],
        (int)X[sample][1],
        output);
    }
return 0; 
}