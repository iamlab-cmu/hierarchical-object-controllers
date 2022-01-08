from distutils.core import setup


setup(name='object_axes_ctrlrs',
      version='1.0.0',
      install_requires=["autolab_core", 
                        "numpy-quaternion", 
                        "gym", 
                        "carbongym", 
                        # "stable_baselines_custom",
                        "stable_baselines",
                        "carbongym-utils",
                        "pytest"],
      description='Compose object centric controllers for RL.',
      url='none',
     )

