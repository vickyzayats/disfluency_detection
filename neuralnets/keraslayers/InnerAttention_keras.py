from keras import backend as K
from keras.engine import Layer
import keras.activations as Activation
from keras.layers import *
import pdb
import numpy as np

class SingleMultiModalAttention(Layer):
    
    def __init__(self, window_size=10, num_feat=20, activation='tanh',
                 att_type='sim', att_norm = 'uniform', backwards=False,
                 position_emb_size=10, **kwargs):
        super(SingleMultiModalAttention, self).__init__(**kwargs)
        self.window = window_size
        self.activation = activation
        self.att_type = att_type
        self.backwards = backwards
        self.num_feat = num_feat
        self.att_norm = att_norm
        self.position_emb_size = position_emb_size
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.dummy_hidden = self.add_weight(name="dummy_hidden",
				 shape=(input_shape[2],),
                                 initializer='zero',
                                 trainable=False)
        if self.att_type in ('sum', 'sim'):
            self.A = self.add_weight(name="A",
				 shape=(input_shape[2], input_shape[2], self.num_feat),
                                 initializer=self.att_norm,
                                 trainable=True)
            self.B = self.add_weight(name="B",
				 shape=(input_shape[2], input_shape[2], self.num_feat),
                                 initializer=self.att_norm,
                                 trainable=True)
        if self.att_type == 'self_att':
            self.W = self.add_weight(name='W',
				 shape=(input_shape[2] + self.position_emb_size, self.num_feat),
                                 initializer=self.att_norm,
                                 trainable=True)
            self.PE = self.add_weight(name='PE',
				 shape=(self.window*2+1, self.position_emb_size),
                                 initializer=self.att_norm,
                                 trainable=True)
            
        super(SingleMultiModalAttention, self).build(input_shape)  # Be sure to call this somewhere!

                                                
    def call(self, x, mask=None):
        padding = K.repeat_elements(K.expand_dims(self.dummy_hidden, axis=0), self.window, axis=0)
	padding = RepeatVector(K.shape(x)[0])(padding) 
	padding = K.permute_dimensions(padding, (1,0,2)) #output_shape=(batch, self.window, x.shape[-1])

        if self.backwards:
            h_pad = K.concatenate([padding, x], axis=1)
	    h_pad = h_pad[:,::-1,:]
	    x = x[:,::-1,:]
        else:
            h_pad = K.concatenate([x, padding], axis=1)
       
	sim = []
	for ii in range(self.window):
	    h_a = h_pad[:,ii:-self.window+ii]
	    h_a = K.dot(h_a, self.A)
	    h_a = K.l2_normalize(h_a, axis=-2)	
	    h_b = K.dot(x, self.B)	
 	    h_b = K.l2_normalize(h_b, axis=-2)
	    sim.append(K.mean(h_a * h_b, axis=-2))
	sim = K.stack(sim, axis=2)
	if self.backwards:
	    sim = sim[:,::-1]
	return sim

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.window, self.num_feat)

    def get_config(self):
        config = {'window_size': self.window, 'activation': self.activation, 'att_type': self.att_type,
                  'backwards': self.backwards, 'num_feat': self.num_feat}
        base_config = super(SingleMultiModalAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class FlattenCNN(Layer):
    
    def __init__(self, **kwargs):
        super(FlattenCNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = True
        super(FlattenCNN, self).build(input_shape)
        
    def call(self, x, mask=None):
	shapes = K.shape(x)
        return K.reshape(x, [shapes[0],shapes[1], -1])
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2] * input_shape[3])



class SelfAttention(Layer):
    
    def __init__(self, num_heads=10, **kwargs):
        self.num_heads = num_heads
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='uniform',
                                 trainable=True)
        self.A = self.add_weight(shape=(input_shape[-1], self.num_heads),
                                 initializer='uniform',
                                 trainable=True)
        super(SelfAttention, self).build(input_shape)
        
    def call(self, x, mask=None):
        h = Activation.relu(K.dot(x, self.W)) # size (batch, seq_len, phone_len, h_dim)
        h = K.dot(h, self.A) # size (batch, seq_len, phone_len, num_heads)
        att = K.permute_dimensions(h, (0,1,3,2))
        att = self.softmax(att)
        att = K.permute_dimensions(att, (0,1,3,2))
        return att
       
    def softmax(self, x, axis=-1):
        xm = x.max(axis=axis, keepdims=True)
        return K.exp(x - xm) / K.exp(x - xm).sum(axis=axis, keepdims=True)
        
    def calculate_output_shape(self, input_shape):
        return tuple(list(input_shape[:-1]) + [self.num_heads])


def attention_custom_object():
    '''Returns the custom objects, needed for loading a persisted model.'''
    instanceHolder = {'instance': None}

    class ClassWrapper(SingleMultiModalAttention):
        def __init__(self, *args, **kwargs):
            instanceHolder['instance'] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)

    return {'SingleMultiModalAttention': ClassWrapper} 

if __name__ == '__main__':
    from keras.models import Sequential
    from keras.layers import Embedding, TimeDistributed, Dense, Flatten, LSTM
    import numpy as np
    vocab_size = 20
    batch_size, maxlen = 32, 40
    n_hiddens = 10
    n_classes = 5
    x = np.random.randint(vocab_size, size=(batch_size, maxlen))
    model = Sequential()
    model.add(Embedding(vocab_size, n_hiddens))
    layer = MyLayer(6)
    model.add(layer)
    code.interact(local=locals())
    lstm = LSTM(n_hiddens, return_sequences=True)
    model.add(lstm)
    code.interact(local=locals())

    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    # Train first mini batch
    x = np.random.randint(1, vocab_size, size=(batch_size, maxlen))
    y = np.random.randint(1, size=(batch_size, maxlen,1))
    #y = np.eye(n_classes)[y]
    model.train_on_batch(x, y)
    
    
    
    print(x)
    print(y)
    
