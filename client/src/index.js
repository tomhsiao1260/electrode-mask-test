import * as THREE from 'three'
import Loader from './Loader'
import ViewerCore from './core/ViewerCore'
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min'

let gui

init()

async function init() {
  // renderer setup
  const canvas = document.querySelector('.webgl')
  const renderer = new THREE.WebGLRenderer({ antialias: true, canvas: canvas })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(window.innerWidth, window.innerHeight)
  renderer.setClearColor(0, 0)
  renderer.outputColorSpace = THREE.SRGBColorSpace

  // const meta = await Loader.getMeta()
  const meta = {}
  const viewer = new ViewerCore({ meta, renderer, canvas })
  update(viewer)
}

function update(viewer) {
  viewer.render()
  updateGUI(viewer)
}

function updateGUI(viewer) {
  if (gui) gui.destroy()
  gui = new GUI()

  const segment = gui.addFolder('segment')
  const slice = gui.addFolder('slice')
  const label = gui.addFolder('label')

  segment.add(viewer.params, 'segmentVisible').name('visible').onChange(viewer.render)
  segment.add(viewer.params, 'colorful', true).name('color').onChange(viewer.render)
  segment.add(viewer.params, 'surface', 0.002, 0.7).name('surface').onChange(viewer.render)

  slice.add(viewer.params, 'sliceVisible').name('visible').onChange(viewer.render)
  slice.add(viewer.params, 'sliceX', -0.5, 0.5).name('x').onChange(viewer.render)
  slice.add(viewer.params, 'sliceY', -0.5, 0.5).name('y').onChange(viewer.render)
  slice.add(viewer.params, 'sliceZ', -0.5, 0.5).name('z').onChange(viewer.render)

  label.add(viewer.params, 'label', 0, 1).name('label').onChange(viewer.render)
  label.add(viewer.params, 'tlabel', 0.01, 0.06).name('thickness').onChange(viewer.render)
}
