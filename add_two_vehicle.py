#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time
random.seed(41)


def main():
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.get_world()

        blueprint_library = world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        bp = random.choice(blueprint_library.filter('vehicle'))

        # A blueprint contains the list of attributes that define a vehicle's
        # instance, we can read them and modify some of them. For instance,
        # let's randomize its color.
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # Now we need to give an initial transform to the vehicle. We choose a
        # random transform from the list of recommended spawn points of the map.
        transform = random.choice(world.get_map().get_spawn_points())
        distance = 10.0
        
        #rear_transform = calculate_transformer(reference_transform, -distance)
        print(transform)
        
        # So let's tell the world to spawn the vehicle.
        vehicle = world.spawn_actor(bp, transform)
        actor_list.append(vehicle)
        time.sleep(2)
        location = vehicle.get_location()
        location.x -= 20
        location.z += 2
        #vehicle.set_location(location)
        transform = vehicle.get_transform()
        transform.location = location
        vehicle_two = world.spawn_actor(bp, transform)
        #vehicle.set_autopilot(True)
        actor_list.append(vehicle_two)
        print('created %s' % vehicle.type_id)
        

        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
        lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)
        actor_list.append(lidar)
        
        lidar.listen(lambda point_cloud: point_cloud.save_to_disk('tutorial/new_lidar_output_one/%.6d.ply' % point_cloud.frame))
        # Second Lidar
        lidar_two_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_two_bp.set_attribute('channels', '32')
        lidar_two_bp.set_attribute('range', '100')
        lidar_two_bp.set_attribute('points_per_second', '100000')
        lidar_two_bp.set_attribute('rotation_frequency', '10')
        lidar_two_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
        lidar = world.spawn_actor(lidar_two_bp, lidar_two_tf, attach_to=vehicle_two)
        actor_list.append(lidar)
        
        lidar.listen(lambda point_cloud: point_cloud.save_to_disk('tutorial/new_lidar_output_two/%.6d.ply' % point_cloud.frame))
        
        time.sleep(50)
        

    finally:

        print('destroying actors')
        #camera.destroy()
        lidar.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':

    main()
