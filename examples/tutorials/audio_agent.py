# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime

import numpy as np
import quaternion as qt
from numpy import ndarray

import habitat_sim
import habitat_sim._ext.habitat_sim_bindings as hsim_bindings
import habitat_sim.sensor
import habitat_sim.sim

import glob
import os
import librosa
import matplotlib.pyplot as plt
from librosa.display import waveplot

def printTime():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

def plotIR(path:str):
    # material_rir, _ = librosa.load('/home/sangarg/AudioSimulation0/ir-SimplificationOff-WithMat.wav', sr=44100, mono=False)
    # no_material_rir, _ = librosa.load('/home/sangarg/AudioSimulation0/ir-simplicationOff-WithoutMat.wav', sr=44100, mono=False)
    rir,_ = librosa.load('/home/sangarg/AudioSimulation0/ir.wav', sr=44100, mono=False)
    no_simplify_rir,_ = librosa.load('/home/sangarg/MeridianSimulation0/ir.wav', sr=44100, mono=False)

    fig, axes = plt.subplots(2, 1)
    max_lenth = 2000
    waveplot(rir[0, :4000], 44100, ax=axes[0])
    waveplot(no_simplify_rir[0, :4000], 44100, ax=axes[1])
    # waveplot(material_rir[0, :4000], 44100, ax=axes[0])
    # waveplot(no_material_rir[0, :4000], 44100, ax=axes[1])
    plt.show()


def runSimulation(path:str):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = (
        "data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
    )
    backend_cfg.scene_dataset_config_file = (
        "data/scene_datasets/mp3d_example/mp3d.scene_dataset_config.json"
    )
    backend_cfg.enable_physics = False

    agent_config = habitat_sim.AgentConfiguration()

    cfg = habitat_sim.Configuration(backend_cfg, [agent_config])

    sim = habitat_sim.Simulator(cfg)

    # create the acoustic configs
    acoustics_config = hsim_bindings.HabitatAcousticsConfiguration()
    acoustics_config.dumpWaveFiles = True
    acoustics_config.enableMaterials = True
    acoustics_config.writeIrToFile = True

    # create channel layout
    channel_layout = hsim_bindings.HabitatAcousticsChannelLayout()
    channel_layout.channelType = (
        hsim_bindings.HabitatAcousticsChannelLayoutType.Binaural
    )
    channel_layout.channelCount = 2

    # create the Audio sensor specs
    audio_sensor_spec = habitat_sim.AudioSensorSpec()
    audio_sensor_spec.uuid = "audio_sensor"
    audio_sensor_spec.outputDirectory = path
    audio_sensor_spec.acousticsConfig = acoustics_config
    audio_sensor_spec.channelLayout = channel_layout

    # add the audio sensor
    sim.add_sensor(audio_sensor_spec)

    # Get the audio sensor object
    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]

    # set audio source location, no need to set the agent location, will be set implicitly
    audio_sensor.setAudioSourceTransform(np.array([-10.3, 1.52, 1.73]))

    # run the simulation
    for i in range(1):
        print(i)
        print("Start Time : ")
        printTime()
        obs = sim.get_sensor_observations()["audio_sensor"]

        # print the audio observations
        # print(obs)

        # write the observations to a file
        p = audio_sensor_spec.outputDirectory + str(i) + "/ir";

        for channelIndex in range (0, len(obs)):
            filePath = p + str(channelIndex) + ".txt"
            f = open(filePath, "w")
            print("Writing file : ", filePath)
            for sampleIndex in range (0, len(obs[channelIndex])):
                f.write(str(sampleIndex) + "\t" + str(obs[channelIndex][sampleIndex]) + "\n")
            f.close()

        print("End Time : ")

        # plotIR(p)

        printTime()

    sim.close()

def main():
    runSimulation('/home/sangarg/AudioSimulation')
    runSimulation('/home/sangarg/MeridianSimulation')


if __name__ == "__main__":
    main()
