import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        
        # Initialize storage for forward pass
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.out = None
        
        # Initialize gradients storage
        self.gradients = {'W1': np.zeros_like(self.W1), 
                         'W2': np.zeros_like(self.W2)}

    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        self.z1 = np.dot(X, self.W1) + self.b1
        
        # Apply activation function
        if self.activation_fn == 'tanh':
            self.a1 = np.tanh(self.z1)
        elif self.activation_fn == 'relu':
            self.a1 = np.maximum(0, self.z1)
        else:  # sigmoid
            self.a1 = 1 / (1 + np.exp(-self.z1))
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.out = np.tanh(self.z2)
        
        return self.out

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        
        if self.activation_fn == 'tanh':
            dz1 = 1 - self.a1**2
        elif self.activation_fn == 'relu':
            dz1 = (self.z1 > 0).astype(float)
        else:  # sigmoid
            dz1 = self.a1 * (1 - self.a1)

        # TODO: update weights with gradient descent
        
        m = X.shape[0]
        delta2 = (self.out - y) * (1 - self.out**2)
        self.gradients['W2'] = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        
        
        delta1 = np.dot(delta2, self.W2.T) * dz1
        self.gradients['W1'] = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m

        # TODO: store gradients for visualization
        
        self.W2 -= self.lr * self.gradients['W2']
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * self.gradients['W1']
        self.b1 -= self.lr * db1

    

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.a1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title('Hidden Layer Features')

    # TODO: Hyperplane visualization in the hidden space

    # TODO: Distorted input space transformed by the hidden layer

    # TODO: Plot input layer decision boundary
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax_input.contourf(xx, yy, Z, cmap='bwr', alpha=0.4)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr')
    ax_input.set_title(f'Input Space at Step {frame}')

    # Network diagram for gradients
    nodes = {
        'x1': (0.0, 0.0),
        'x2': (0.0, 1.0),
        'h1': (0.5, 0.0),
        'h2': (0.5, 0.5),
        'h3': (0.5, 1.0),
        'y': (1.0, 0.0)
    }
    
    for name, pos in nodes.items():
        circle = plt.Circle(pos, 0.05, color='blue')
        ax_gradient.add_patch(circle)
        ax_gradient.text(pos[0]-0.05, pos[1]+0.1, name)
    
    for i in range(2): 
        for j in range(3): 
            weight = abs(mlp.gradients['W1'][i, j])
            ax_gradient.plot([nodes[f'x{i+1}'][0], nodes[f'h{j+1}'][0]],
                           [nodes[f'x{i+1}'][1], nodes[f'h{j+1}'][1]],
                           '-', color='purple', alpha=weight, linewidth=1+3*weight)
    
    for j in range(3):
        weight = abs(mlp.gradients['W2'][j, 0])
        ax_gradient.plot([nodes[f'h{j+1}'][0], nodes['y'][0]],
                        [nodes[f'h{j+1}'][1], nodes['y'][1]],
                        '-', color='purple', alpha=weight, linewidth=1+3*weight)
    
    ax_gradient.set_title(f'Gradients at Step {frame}')
    ax_gradient.set_xlim(-0.2, 1.2)
    ax_gradient.set_ylim(-0.2, 1.2)
    
    ax_hidden.grid(True)
    ax_input.grid(True)
    ax_gradient.grid(True)


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)