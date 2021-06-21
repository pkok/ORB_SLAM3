# Modified by Raul Mur-Artal
# Automatically compute the optimal scale factor for monocular VO/SLAM.

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Requirements: 
# sudo apt-get install python-argparse

"""
This script computes the absolute trajectory error from the ground truth
trajectory and the estimated trajectory.
"""

import sys
import numpy
import scipy.stats
import argparse
import associate
from dataclasses import dataclass, asdict
import os.path

import pandas
import plotly.express as px

COLORS = (
    "rgb(245,   0,  47)", # uva-red
    "rgb(172, 126,  57)", # uva-gold
    "rgb(115,   0, 195)", # uva-fnwi-purple
    "rgb(  0,  72, 232)", # uva-fgw-blue
    "rgb(171, 184,   0)", # uva-feb-green
    "rgb(255, 107,   0)", # uva-fgw-orange
    "rgb(168,   0,  86)", # uva-fdr-red
    "rgb( 28, 255, 227)", # uva-iis-blue
    "rgb(255,  13,   0)", # uva-ilo-red
)

def plotly_histogram(data, statistics, title):
    import plotly.graph_objects as go
    fig = go.Figure()
    for column in data.columns:
        fig.add_trace(go.Histogram(x=data[column]))

    means = statistics['mean']
    for i, mean in enumerate(means):
        fig.add_vline(x=mean,
                      annotation_text=f"mean {i}")

    fig.update_layout(
        barmode='overlay',
        title_text=title,
        xaxis_title_text="Absolute translational error",
        yaxis_title_text="Frequency"
    )
    fig.update_traces(opacity=0.75)
    fig.show(renderer="browser")


def plotly_violin(data, output_file):
    import plotly.graph_objects as go
    import plotly.io

    full_path = os.path.abspath(output_file)
    target_dir = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    extension_start = filename.rfind(os.path.extsep)
    file_base = filename[:extension_start]
    extension = filename[extension_start+1:]
    if extension not in plotly.io.renderers:
        raise RuntimeError(f"Can't generate a plot of type '{extension}'")
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    filename = (target_dir + os.path.sep + 
                file_base + "-violin" +
                os.path.extsep + extension)

    template = plotly.io.templates['plotly_white']
    template.layout['colorway'] = COLORS
    layout = go.Layout(autosize=False,
                       template=template,
                       legend_traceorder="reversed",
                       width=1400,
                       height=1000)
    fig = go.Figure(layout=layout)
    for i, column in enumerate(reversed(data.columns)):
        opaque_color = COLORS[len(data.columns) - 1 - i]
        transparant_color = "rgba(" + opaque_color[4:-1] + ", .3)"
        fig.add_trace(go.Violin(x=data[column],
                                name=column,
                                legendgroup=column,
                                scalegroup=column,
                                fillcolor=transparant_color,
                                marker=go.violin.Marker(color=opaque_color,
                                                        opacity=0.1),
                                box_visible=True, 
                                meanline_visible=True,
                                showlegend=True,
                                points="all"))

    fig.update_layout(violingap=0, violinmode='overlay')
    fig.write_image(filename)


def normalize_data(data):
    stats = data.describe().transpose()
    mean = stats['mean']
    scale = stats['75%'] - stats['25%']
    return (data - mean) / scale 


def align(model,data):
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    """


    numpy.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)
    
    W = numpy.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity( 3 ))
    if(numpy.linalg.det(U) * numpy.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh

    rotmodel = rot*model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += numpy.dot(data_zerocentered[:,column].transpose(),rotmodel[:,column])
        normi = numpy.linalg.norm(model_zerocentered[:,column])
        norms += normi*normi

    s = float(dots/norms)    
    
    transGT = data.mean(1) - s*rot * model.mean(1)
    trans = data.mean(1) - rot * model.mean(1)

    model_alignedGT = s*rot * model + transGT
    model_aligned = rot * model + trans

    alignment_errorGT = model_alignedGT - data
    alignment_error = model_aligned - data

    trans_errorGT = numpy.sqrt(numpy.sum(numpy.multiply(alignment_errorGT,alignment_errorGT),0)).A[0]
    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(alignment_error,alignment_error),0)).A[0]
        
    return rot,transGT,trans_errorGT,trans,trans_error, s

def plot_traj(ax,stamps,traj,style,color,label):
    """
    Plot a trajectory using matplotlib. 
    
    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    
    """
    stamps.sort()
    interval = numpy.median([s-t for s,t in zip(stamps[1:],stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x)>0:
            ax.plot(x,y,style,color=color,label=label)
            label=""
            x=[]
            y=[]
        last= stamps[i]
    if len(x)>0:
        ax.plot(x,y,style,color=color,label=label)
            

def main():
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory. 
    ''')
    parser.add_argument('first_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('second_file', nargs="+", help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)',default=1.0)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 10000000 ns)',default=20000000)
    parser.add_argument('--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    parser.add_argument('--save_associations', help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    parser.add_argument('--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    parser.add_argument('--verbose', help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)', action='store_true')
    parser.add_argument('--verbose2', help='print scale eror and RMSE absolute translational error in meters after alignment with and without scale correction', action='store_true')
    parser.add_argument('--export-ate', help='export the aligned ATEs to a csv file')
    parser.add_argument('--import-ate', help='import the aligned ATEs from a csv file', action='store_true')
    args = parser.parse_args()
    errors = []
    stats = []

    errorsGT = []
    statsGT = []

    if args.import_ate:
        pass
    if not args.import_ate:
        for f in args.second_file:
            error, errorGT = evaluate(first_file=args.first_file,
                                      second_file=f,
                                      offset=args.offset, 
                                      scale=args.scale,
                                      max_difference=args.max_difference,
                                      save=args.save,
                                      save_associations=args.save_associations,
                                      plot=args.plot,
                                      verbose=args.verbose,
                                      verbose2=args.verbose2)
            errors.append(error)
            errorsGT.append(errorGT)

    # Convert errors -> pandas.DataFrame
    columns = [f"run {i+1}" for i in range(len(errors))]
    errors = pandas.DataFrame(errors).transpose()
    errorsGT = pandas.DataFrame(errorsGT).transpose()

    errors.columns = columns
    errorsGT.columns = columns

    normalized_errors = normalize_data(errors)

    if args.export_ate:
        errors.to_csv(args.export_ate, index=False)

    if args.plot:
        plotly_violin(errors, args.plot)
        plotly_violin(normalized_errors,
                      "normalized-"+args.plot)

    if args.verbose:
        print("A summary of statistics:")
        print("  - errors")
        print(stats)
        print("\n  - errorsGT")
        print(statsGT)
        print("\n - zero centered errors")
        print(normalized_errors)


def evaluate(first_file, second_file, offset, scale, max_difference,
             save, save_associations, plot, verbose, verbose2):
    first_list = associate.read_file_list(first_file, False)
    second_list = associate.read_file_list(second_file, False)

    matches = associate.associate(first_list, second_list,float(offset),float(max_difference))    
    if len(matches)<2:
        sys.exit("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")
    first_xyz = numpy.matrix([[float(value) for value in first_list[a][0:3]] for a,b in matches]).transpose()
    second_xyz = numpy.matrix([[float(value)*float(scale) for value in second_list[b][0:3]] for a,b in matches]).transpose()
    dictionary_items = list(second_list.items())
    sorted_second_list = sorted(dictionary_items)

    second_xyz_full = numpy.matrix([[float(value)*float(scale) for value in sorted_second_list[i][1][0:3]] for i in range(len(sorted_second_list))]).transpose() # sorted_second_list.keys()]).transpose()
    rot,transGT,trans_errorGT,trans,trans_error, scale = align(second_xyz,first_xyz)

    return trans_error, trans_errorGT


if __name__ == "__main__":
    main()
