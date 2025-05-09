# Use the official TensorFlow Docker image with GPU support
FROM tensorflow/tensorflow:latest-gpu-jupyter

# # Set the working directory inside the container
# WORKDIR /tf/Model

# # Copy all files (adjust as needed)
# COPY . /tf/Model

# Install additional dependencies including librosa and tensorflow-addons
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyterlab \
    librosa \
    tensorflow-addons  # Install tensorflow-addons package

ENV PATH=/usr/local/cuda-12.6/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Default command to run Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]

