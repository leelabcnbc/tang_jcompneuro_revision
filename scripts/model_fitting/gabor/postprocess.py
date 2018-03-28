from tang_jcompneuro.model_fitting_postprocess import handle_one_model_type

if __name__ == '__main__':
    handle_one_model_type('gabor',
                          # lambda x: x.startswith('vgg16_bn+')
                          )
