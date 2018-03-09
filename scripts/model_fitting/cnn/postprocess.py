from tang_jcompneuro.model_fitting_postprocess import handle_one_model_type

if __name__ == '__main__':
    handle_one_model_type('cnn',
                          # use this to filter, if you like to remove some intermediate stuff.
                          # lambda x: x.startswith('b.1/') or x.startswith('b.2/')
                                    # or x.startswith('b.9/') or x.startswith('b.12/')
                          )
