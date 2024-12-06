import { ShaderMaterial } from 'three'

export class GenerateSDFMaterial extends ShaderMaterial {
  constructor(params) {
    super({
      defines: {},

      uniforms: {
        sliceData: { value: null },
      },

      vertexShader: /* glsl */ `
        varying vec2 vUv;

        void main() {
            vUv = uv;
            vUv.y = 1.0 - vUv.y;
            gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
        }
        `,

      fragmentShader: /* glsl */ `
        varying vec2 vUv;
        uniform sampler2D sliceData;

        void main() {
            // gl_FragColor = vec4(vUv.x, 0.0, 0.0, 0.0);
            gl_FragColor = texture2D(sliceData, vUv);
        }
        `,
    })
  }
}
