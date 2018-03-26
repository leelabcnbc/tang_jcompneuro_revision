from tang_jcompneuro.model_fitting_postprocess import handle_one_model_type

if __name__ == '__main__':
    handle_one_model_type('cnnpre',
                          # lambda x: x.startswith('gqm.2_poisson/MkE2_Shape/OT/25')
                          )
