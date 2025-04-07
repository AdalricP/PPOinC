#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define STATE_DIM 4      
#define ACTION_DIM 2     
#define GAMMA 0.99       
#define EPSILON 0.2      
#define LEARNING_RATE 0.001 
#define EPOCHS 3 


typedef struct {
    double weights[STATE_DIM][ACTION_DIM];
    double bias[ACTION_DIM];
} Policy;


void softmax(double *input, double *output, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i]);
        sum += output[i];
    }
    for (int i = 0; i < size; i++)
        output[i] /= sum;
}

void initialise_policy(Policy *policy){
    srand(time(NULL));
    for (int i = 0; i < STATE_DIM; i++) {
        for (int j = 0; j < ACTION_DIM; j++) {
            policy->weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
    for (int j = 0; j < ACTION_DIM; j++)
        policy->bias[j] = 0.0;
}

void get_action_probs(Policy *policy, double *state, double *probs) {
    double logits[ACTION_DIM] = {0};
    for (int j = 0; j < ACTION_DIM; j++) {
        for (int i = 0; i < STATE_DIM; i++)
            logits[j] += state[i] * policy->weights[i][j];
        logits[j] += policy->bias[j];
    }
    softmax(logits, probs, ACTION_DIM);
}

int sample_action(double *probs, int n) {
    double r = (double)rand() / RAND_MAX;
    double cum_sum = 0;
    for (int i = 0; i < n; i++) {
        cum_sum += probs[i];
        if (r < cum_sum)
            return i;
    }
    return n - 1; 
}

void compute_returns(double *rewards, int size, double *returns) {
    double sum = 0;
    for (int i = size - 1; i >= 0; i--) {
        sum = rewards[i] + GAMMA * sum;
        returns[i] = sum;
    }
}

void update_policy(Policy *policy, double *states, int *actions, double *log_probs_old, double *returns, int batch_size) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < batch_size; i++) {
            double probs[ACTION_DIM];
            get_action_probs(policy, &states[i * STATE_DIM], probs);
            double log_prob_new = log(probs[actions[i]]);

            double ratio = exp(log_prob_new - log_probs_old[i]);
            double unclipped = ratio * returns[i];
            double clipped = fmax(fmin(ratio, 1 + EPSILON), 1 - EPSILON) * returns[i];

            double loss = -fmin(unclipped, clipped);  

            for (int j = 0; j < STATE_DIM; j++)
                policy->weights[j][actions[i]] -= LEARNING_RATE * loss * states[i * STATE_DIM + j];
                
            policy->bias[actions[i]] -= LEARNING_RATE * loss;
        }
    }
}

void train_ppo() {
    Policy policy;
    initialise_policy(&policy);

    int batch_size = 1000;
    
    double states[batch_size * STATE_DIM];
    int actions[batch_size];
    double rewards[batch_size];
    double log_probs_old[batch_size];
    double returns[batch_size];

    for (int iter = 0; iter < 1000; iter++) {
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < STATE_DIM; j++)
                states[i * STATE_DIM + j] = ((double)rand() / RAND_MAX) * 2 - 1;

            double probs[ACTION_DIM];
            get_action_probs(&policy, &states[i * STATE_DIM], probs);
            actions[i] = sample_action(probs, ACTION_DIM);
            log_probs_old[i] = log(probs[actions[i]]);
            rewards[i] = (actions[i] == 1) ? 1.0 : -1.0;  
        }

        compute_returns(rewards, batch_size, returns);
        update_policy(&policy, states, actions, log_probs_old, returns, batch_size);

        if (iter % 100 == 0)
            printf("Iteration %d: Training...\n", iter);
    }
}

int main(void){
    train_ppo();
    return 0;
}
