import * as THREE from 'three'
import { NRRDLoader } from 'three/examples/jsm/loaders/NRRDLoader'

export default class Loader {
  constructor() {}

  // static getMeta() {
  //   return fetch('meta.json').then((res) => res.json())
  // }

  static getTexture(filename) {
    return new THREE.TextureLoader().loadAsync(filename)
  }

  static getVolumeData(filename) {
    return new NRRDLoader().loadAsync(filename)
  }
}
