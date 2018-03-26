from tang_jcompneuro.model_fitting_postprocess import handle_one_model_type

if __name__ == '__main__':
    handle_one_model_type('cnnpre',
                          # lambda x: ('+conv5_1/' in x) or ('+fc6/' in x)
                          )
