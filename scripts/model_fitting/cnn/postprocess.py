from tang_jcompneuro.model_fitting_postprocess import handle_one_model_type

if __name__ == '__main__':
    handle_one_model_type('cnn',
                          # use this to filter, if you like to remove some intermediate stuff.
                          # lambda x: x.startswith('b.') and ('_' not in x.split('/')[0]) \
                          #           and x.split('/')[0] not in {'b.4','b.5',
                          #                                       'b.7','b.8',
                          #                                       'b.10','b.11'}
                          # lambda x: x.startswith('b.') and ('_' in x.split('/')[0])
                          lambda x: x.startswith('mlp.')
                          )
