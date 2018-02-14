import click
import numpy as np
from astropy.io import fits
import astropy.units as u


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ..coordinates import get_pixel_coordinates


def gaussian_component(x, y, flux, x_fwhm, y_fwhm, rotation, center=None):

    if center is None:
        x_0 = y_0 = len(x) // 2
    else:
        x_0 = center[0].value
        y_0 = center[1].value

    rotation = np.deg2rad(rotation)

    x = x.value - x_0
    y = y.value - y_0

    x_rot = (np.cos(rotation) * x - np.sin(rotation) * y) + x_0
    y_rot = np.sin(rotation) * x + np.cos(rotation) * y + y_0

    return flux * np.exp(-4*np.log(2) * ((x_rot-x_0)**2 / x_fwhm**2 + (y_rot-y_0)**2 / y_fwhm**2))


def model_from_components(data, header):

    x, y = get_pixel_coordinates(header)
    X, Y = np.meshgrid(x, y)
    list_gauss_mod = []
    centers = []
    list_flux = []
    for comp in data:
        flux = comp[0]
        center = [(comp[1] * u.deg).to(u.mas), (comp[2] * u.deg).to(u.mas)]
        x_fwhm = (comp[3] * u.deg).to(u.mas)
        y_fwhm = (comp[4] * u.deg).to(u.mas)
        rotation = comp[5] * u.deg

        list_flux.append(flux)
        centers.append([(comp[1] * u.deg).to(u.mas).value, (comp[2] * u.deg).to(u.mas).value])
        list_gauss_mod.append(gaussian_component(X, Y, flux, x_fwhm.value, y_fwhm.value, rotation.value, center=center))

    return np.asarray(list_gauss_mod).sum(axis=0), centers, list_flux


@click.command()
@click.argument('input_file', type=click.Path(file_okay=True, dir_okay=False))
def main(
    input_file,
):
    difmap_data = fits.open(input_file)
    gaussian_components_data = difmap_data['AIPS CC'].data
    clean_map_header = difmap_data['PRIMARY'].header

    model, centers, flux = model_from_components(gaussian_components_data, clean_map_header)
    clean_map = difmap_data['PRIMARY'].data[0][0]

    residuals = clean_map - model

    x, y = get_pixel_coordinates(clean_map_header)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.pcolormesh(x.value, y.value, residuals)
    ax.scatter(np.asarray(centers)[:, 0], np.asarray(centers)[:, 1], c=flux, marker='+')
    ax.axis([-x.min().value, -x.max().value, y.min().value, y.max().value])
    ax.set_aspect(1)
    plt.show()


if __name__ == '__main__':
    main()
