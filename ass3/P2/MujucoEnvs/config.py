
'''
This file contains the configs used for Model creation and training. You need to give your best hyperparameters and the configs you used to get the best results for 
every environment and experiment.  These configs will be automatically loaded and used to create and train your model in our servers.
'''
#You can add extra keys or modify to the values of the existing keys in bottom level of the dictionary.
#DONT CHANGE THE OVERALL STRUCTURE OF THE DICTIONARY. 

configs = {
    
    
    'InvertedPendulum-v4': {
        
        "RL":{
            #You can add or change the keys here
           "hyperparameters": {
                'hidden_size': ,
                'n_layers': ,
                'activation': ,
                'out_activation':  ,
                'lr': ,
                'ntraj' : ,
                'max_length':,
                'std_dev_max' :,
                'std_dev_min' :,
                'std_dev_DF' : ,
                'std_dev_K' : ,
                'gamma' : ,
                'AvgRewBaseline' : False
                
            },
            "num_iteration": ,
        },

        "AC":{
            #You can add or change the keys here
           "hyperparameters": {
                'critic_hidden_size': ,
                'critic_activation' :  ,
                'critic_n_layers' : ,
                'hidden_size': ,
                'n_layers': ,
                'activation': ,
                'output_scaling_factor' : ,
                'out_activation': 'tanh' ,
                'lr': ,
                'ntraj' : ,
                'max_length':,
                'std_dev_max' :,
                'std_dev_min' :,
                'std_dev_DF' : ,
                'std_dev_K' :,
                'gamma' : 
                
            },
            "num_iteration": ,
        },
        
         "SB3":{
            #You can add or change the keys here
                "hyperparameters": {
                    "total_timesteps": ,
                    "algorithm": ,
                    "log_interval" :,
                    "eval_freq" :,
                    "save_freq" :,
                    "Run_name" :'Run',
                    
                },            
        },
    },
    
    'HalfCheetah-v4': {
        
        "RL":{
            #You can add or change the keys here
               "hyperparameters": {
                'hidden_size': ,
                'n_layers':,
                'activation': ,
                'output_scaling_factor' :,
                'out_activation': ,
                'lr': ,
                'ntraj' : ,
                'max_length': ,
                'std_dev_max' : ,
                'std_dev_min' : ,
                'std_dev_DF' :  ,
                'std_dev_K' :  ,
                'gamma' :  ,
                'AvgRewBaseline' :  False               
            },
            "num_iteration": ,
        },
        
        "AC":{
            #You can add or change the keys here
           "hyperparameters": {
                'critic_hidden_size': ,
                'critic_activation' :  ,
                'critic_n_layers' :  ,
                'hidden_size': ,
                'n_layers': ,
                'activation':,
                'output_scaling_factor' :,
                'out_activation':  ,
                'lr': ,
                'ntraj' : ,
                'max_length':,
                'std_dev_max' :,
                'std_dev_min' :,
                'std_dev_DF' : ,
                'std_dev_K' :,
                'gamma' : 
                
            },
            "num_iteration": ,
        },
        
         "SB3":{
            #You can add or change the keys here
                "hyperparameters": {
                    "total_timesteps": ,
                    "algorithm": 'SAC',
                    "log_interval" :,
                    "eval_freq" :,
                    "save_freq" :,
                    "Run_name" :,
              
                },            
        },
    },
    
    

    'Hopper-v4': {
        
        "RL":{
            #You can add or change the keys here
               "hyperparameters": {
                'hidden_size': ,
                'n_layers': ,
                'activation':,
                'output_scaling_factor' :,
                'out_activation': ,
                'lr':  ,
                'ntraj' :  ,
                'max_length': ,
                'std_dev_max' : ,   
                'std_dev_min' : ,
                'std_dev_DF' :  ,
                'std_dev_K' : ,
                'gamma' : ,
                'AvgRewBaseline' : True
                
            },
            "num_iteration": ,
        },
        
        "AC":{
            #You can add or change the keys here
           "hyperparameters": {
                'critic_hidden_size': ,
                'critic_activation' :  ,
                'critic_n_layers' :  ,
                'hidden_size': ,
                'n_layers': ,
                'activation': ,
                'output_scaling_factor' :,
                'out_activation':,
                'lr': ,
                'ntraj' : ,
                'max_length': ,
                'std_dev_max' : ,
                'std_dev_min' : ,
                'std_dev_DF' :  ,
                'std_dev_K' : ,
                'gamma' :
                
            },
            "num_iteration":,
        },
        
       "SB3":{
            #You can add or change the keys here
                "hyperparameters": {
                    "total_timesteps":,
                    "algorithm": 'SAC',
                    "log_interval" :,
                    "eval_freq" :,
                    "save_freq" :,
                    "Run_name" :,
            
                },            
        },
    },
    
    