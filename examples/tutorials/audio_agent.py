# # Copyright (c) Facebook, Inc. and its affiliates.
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

from datetime import datetime

import glob
import os
import librosa
import matplotlib.pyplot as plt
from librosa.display import waveplot
from habitat_sim.utils.common import quat_from_angle_axis

import numpy as np
import quaternion as qt
from numpy import ndarray
from scipy.io import wavfile

import habitat_sim
import habitat_sim._ext.habitat_sim_bindings as hsim_bindings
import habitat_sim.sensor
import habitat_sim.sim


def printTime():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

def plotIR(path:str):
    # material_rir, _ = librosa.load('/home/sangarg/AudioSimulation0/ir-SimplificationOff-WithMat.wav', sr=44100, mono=False)
    # no_material_rir, _ = librosa.load('/home/sangarg/AudioSimulation0/ir-simplicationOff-WithoutMat.wav', sr=44100, mono=False)
    rir,_ = librosa.load('/home/sangarg/AudioSimulation0/ir.wav', sr=16000, mono=False)
    no_simplify_rir,_ = librosa.load('/home/sangarg/AudioSimulation0/ir-old.wav', sr=16000, mono=False)

    fig, axes = plt.subplots(2, 1)
    max_lenth = 2000
    waveplot(rir[0, :1000], 16000, ax=axes[0])
    waveplot(no_simplify_rir[0, :1000], 16000, ax=axes[1])
    # waveplot(material_rir[0, :4000], 44100, ax=axes[0])
    # waveplot(no_material_rir[0, :4000], 44100, ax=axes[1])
    plt.show()

def main():
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = (
        "17DRP5sb8fy.glb"
    )
    backend_cfg.scene_dataset_config_file = (
        "data/scene_datasets/mp3d_example/mp3d.scene_dataset_config.json"
    )
    backend_cfg.enable_physics = False

    agent_config = habitat_sim.AgentConfiguration()

    cfg = habitat_sim.Configuration(backend_cfg, [agent_config])

    sim = habitat_sim.Simulator(cfg)
    agent = sim.get_agent(0)
    new_state = sim.get_agent(0).get_state()
    new_state.position = np.array([-10.308625, -0.00842199999999993, -1.269901])
    # new_state.position = np.array([-10.3, 0.02, 1.73])
    new_state.rotation = quat_from_angle_axis(0, np.array([0, 1, 0]))
    new_state.sensor_states = {}
    agent.set_state(new_state, True)

    # create the acoustic configs
    acoustics_config = hsim_bindings.HabitatAcousticsConfiguration()
    acoustics_config.dumpWaveFiles = True
    # acoustics_config.enableMaterials = True
    acoustics_config.writeIrToFile = True
    # acoustics_config.diffraction = True
    # acoustics_config.transmission = False
    # acoustics_config.globalVolume = 8
    # acoustics_config.meshSimplification = False
    # acoustics_config.sampleRate = 16000

    # create channel layout
    channel_layout = hsim_bindings.HabitatAcousticsChannelLayout()
    channel_layout.channelType = (
        hsim_bindings.HabitatAcousticsChannelLayoutType.Binaural
    )
    channel_layout.channelCount = 2

    # create the Audio sensor specs
    audio_sensor_spec = habitat_sim.AudioSensorSpec()
    audio_sensor_spec.uuid = "audio_sensor"
    audio_sensor_spec.outputDirectory = "/home/sangarg/AudioSimulation"
    audio_sensor_spec.acousticsConfig = acoustics_config
    audio_sensor_spec.channelLayout = channel_layout

    # add the audio sensor
    sim.add_sensor(audio_sensor_spec)

    # Get the audio sensor object
    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]

    # set audio source location, no need to set the agent location, will be set implicitly
    # audio_sensor.setAudioSourceTransform(np.array([-10.308625, 1.521203000000000083, 0.730099]))
    audio_sensor.setAudioSourceTransform(np.array([-10.308625, 1.521203000000000083, 0.730099]))

    # Mn::Vector3 g_SourcePos = {-10.308625, 0.021203000000000083, 0.730099};
    # Mn::Vector3 g_AgentPos = {-10.308625, -0.00842199999999993, -1.269901};

    # audio_sensor.setAudioSourceTransform(np.array([-9.308625, 1.41578, -4.2699]))
    # Mn::Vector3 g_SourcePos = {-10.308625, 0.021203000000000083, 0.730099};
    # Mn::Vector3 g_AgentPos = {-9.308625, -0.00842199999999993, -4.2699};
    # run the simulation
    for i in range(1):
        print(i)
        print("Start Time : ")
        printTime()
        # obs = sim.get_sensor_observations()["audio_sensor"]

        obs = np.transpose(np.array(sim.get_sensor_observations()["audio_sensor"]))
        print(obs.shape)
        wavfile.write('no_material.wav', 16000, obs)

        # for channelIndex in range (0, len(obs)):
        #     filePath = p + str(channelIndex) + ".txt"
        #     f = open(filePath, "w")
        #     print("Writing file : ", filePath)
        #     for sampleIndex in range (0, len(obs[channelIndex])):
        #         f.write(str(sampleIndex) + "\t" + str(obs[channelIndex][sampleIndex]) + "\n")
        #     f.close()

        # plotIR(p)

        print("End Time : ")
        printTime()

    sim.close()


if __name__ == "__main__":
    main()
