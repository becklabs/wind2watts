
# wind2watts
 Cycling power is a critical measure of effort during training. It is far more robust than heart rate, which tends to lag and doesn't fully capture the instantaneous demands of cycling. However, devices that measure power, such as pedal and crank-based systems, are expensive and often a barrier for many athletes looking for fine-grained control over their training.

[Strava](https://strava.com) offers a power prediction but it has its limitations. Strava assumes 0 m/s wind speed and predicts a distribution of power, rather than a more valuable time-series format.

To address these issues, **wind2watts** employs historical wind data, coupled with modern sequence modeling techniques to deliver superior power prediction.


## Setup

To get your Python environment set up and start working with `wind2watts`:

1. Clone the repository:
```bash
git clone https://github.com/username/wind2watts.git && cd wind2watts
```

2. Create a virtual environment and install the `wind2watts` module and its dependencies:
```bash
python -m venv venv
source venv/bin/activate
python -m pip install -e .
```

3. Install `PyTorch`

For non-CUDA version of PyTorch, use:

```bash
python -m pip install torch==1.10.2
```
For CUDA version of PyTorch, replace `<cuda_version>` with your CUDA version (e.g., `cu101`, `cu102`, `cu110`, etc.):

```bash
python -m pip install torch==1.10.2+<cuda_version> -f https://download.pytorch.org/whl/torch_stable.html
```

Note: To check your CUDA version, use the command `nvcc --version`. 

## Roadmap

The project is under active development, and we have exciting plans for its future. Here's a snapshot of what's coming:

- [ ] Architecture benchmarking: Comparison of different model architectures to optimize performance.
- [ ] Incorporate rider height and weight: Including these physiological factors in the model, which significantly affect aerodynamics and power on gradients, will increase accurate generalization across athletes.
- [ ] Strava App Integration: Implement a web integration that adds power prediction data to subscribers' activity descriptions when they are uploaded.
- [ ] Real-time inference on Garmin/Wahoo: Expand the utility of the model by offering real-time power prediction on Garmin and Wahoo devices.

## Contributions

Contributions are welcome!

## License

This project falls under the MIT license. For additional details, please refer to the [License file](LICENSE.md).