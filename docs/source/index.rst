NGSidekick
==========

Tools for neuroglancer scenes.

.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   state_utils
   segmentprops
   annotations
   segmentcolors
   gcs
   ngvideo_helper


Installation
------------

Packages are available from both PyPI and conda-forge.

.. code-block:: bash

   pip install ngsidekick


.. code-block:: bash

   conda install -c conda-forge ngsidekick


For additional features:

.. code-block:: bash

   pip install ngsidekick[gcs]  # For Google Cloud Storage support


.. code-block:: bash

   conda install -c conda-forge ngsidekick google-cloud-storage
