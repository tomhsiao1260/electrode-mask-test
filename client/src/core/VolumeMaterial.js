import { ShaderMaterial, Matrix4, Vector2, Vector3 } from 'three'

export class VolumeMaterial extends ShaderMaterial {
  constructor(params) {
    super({
      defines: {
        // The maximum distance through our rendering volume is sqrt(3).
        MAX_STEPS: 887, // 887 for 512^3, 1774 for 1024^3
        SURFACE_EPSILON: 0.001,
      },

      uniforms: {
        cmdata: { value: null },
        labelTex: { value: null },
        sdfTex: { value: null },
        volumeTex: { value: null },
        clim: { value: new Vector2(0.4, 1.0) },
        size: { value: new Vector3() },
        projectionInverse: { value: new Matrix4() },
        sdfTransformInverse: { value: new Matrix4() },
        segmentVisible: { value: true },
        sliceVisible: { value: true },
        labelVisible: { value: true },
        slice: { value: new Vector3() },
        colorful: { value: true },
        surface: { value: 0 },
        label: { value: 0 },
        tlabel: { value: 0 },
      },

      vertexShader: /* glsl */ `
        varying vec2 vUv;

        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
        }
        `,

      fragmentShader: /* glsl */ `
        precision highp sampler3D;

        uniform bool segmentVisible;
        uniform bool sliceVisible;
        uniform bool labelVisible;

        varying vec2 vUv;
        uniform vec2 clim;
        uniform vec3 size;
        uniform vec3 slice;
        uniform bool colorful;
        uniform float label;
        uniform float tlabel;
        uniform float surface;
        uniform sampler3D labelTex;
        uniform sampler3D sdfTex;
        uniform sampler3D volumeTex;
        uniform sampler2D cmdata;
        uniform mat4 projectionInverse;
        uniform mat4 sdfTransformInverse;

        const float relative_step_size = 1.0;
        const vec4 ambient_color = vec4(0.2, 0.4, 0.2, 1.0);
        const vec4 diffuse_color = vec4(0.8, 0.2, 0.2, 1.0);
        const vec4 specular_color = vec4(1.0, 1.0, 1.0, 1.0);
        const float shininess = 40.0;

        vec4 cast_mip(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray);

        vec4 apply_colormap(float val);
        vec4 add_lighting(float val, vec3 loc, vec3 step, vec3 view_ray);
        void segment(inout vec3 p, inout bool hit, float surface, vec3 rayDir);

        // distance to box bounds
				vec2 rayBoxDist( vec3 boundsMin, vec3 boundsMax, vec3 rayOrigin, vec3 rayDir ) {
					vec3 t0 = ( boundsMin - rayOrigin ) / rayDir;
					vec3 t1 = ( boundsMax - rayOrigin ) / rayDir;
					vec3 tmin = min( t0, t1 );
					vec3 tmax = max( t0, t1 );
					float distA = max( max( tmin.x, tmin.y ), tmin.z );
					float distB = min( tmax.x, min( tmax.y, tmax.z ) );
					float distToBox = max( 0.0, distA );
					float distInsideBox = max( 0.0, distB - distToBox );
					return vec2( distToBox, distInsideBox );
				}

        // slice XYZ plane
        void sliceXYZ(out vec3 p, out bool hit, vec3 center, vec3 rayOrigin, vec3 rayDirection) {
          float gap = -0.001;
          vec2 boxInfoZ = rayBoxDist( vec3(-0.5, -0.5, center.z - gap), vec3(0.5, 0.5, center.z + gap), rayOrigin, rayDirection );
          vec2 boxInfoY = rayBoxDist( vec3(-0.5, center.y - gap, -0.5), vec3(0.5, center.y + gap, 0.5), rayOrigin, rayDirection );
          vec2 boxInfoX = rayBoxDist( vec3(center.x - gap, -0.5, -0.5), vec3(center.x + gap, 0.5, 0.5), rayOrigin, rayDirection );

          float t = 1e5;
          if (boxInfoZ.y > 0.0) { t = boxInfoZ.x; }
          if (boxInfoY.y > 0.0 && t > boxInfoY.x) { t = boxInfoY.x; }
          if (boxInfoX.y > 0.0 && t > boxInfoX.x) { t = boxInfoX.x; }

          if (t < 1e5) { hit = true; p = rayOrigin + rayDirection * ( t + 1e-5 ); return;  }
          hit = false;
        }

        void main() {
          float fragCoordZ = -1.;

          // float v = texture(volumeTex, vec3( vUv, 0.0 )).r;
          // gl_FragColor = vec4(v, v, v, 1.0); return;

          // get the inverse of the sdf box transform
					mat4 sdfTransform = inverse( sdfTransformInverse );
          // convert the uv to clip space for ray transformation
					vec2 clipSpace = 2.0 * vUv - vec2( 1.0 );
          // get world ray direction
          vec3 rayOrigin = vec3( 0.0 );
          vec4 homogenousDirection = projectionInverse * vec4( clipSpace, - 1.0, 1.0 );
          vec3 rayDirection = normalize( homogenousDirection.xyz / homogenousDirection.w );
          // transform ray into local coordinates of sdf bounds
          vec3 sdfRayOrigin = ( sdfTransformInverse * vec4( rayOrigin, 1.0 ) ).xyz;
          vec3 sdfRayDirection = normalize( ( sdfTransformInverse * vec4( rayDirection, 0.0 ) ).xyz );
          // find whether our ray hits the box bounds in the local box space
          vec2 boxIntersectionInfo = rayBoxDist( vec3( - 0.5 ), vec3( 0.5 ), sdfRayOrigin, sdfRayDirection );
          float distToBox = boxIntersectionInfo.x;
          float distInsideBox = boxIntersectionInfo.y;
					bool intersectsBox = distInsideBox > 0.0;
					gl_FragColor = vec4( 0.0 );

          if ( intersectsBox ) {
            int nsteps = int(boxIntersectionInfo.y * size.x / relative_step_size + 0.5);
            if ( nsteps < 1 ) discard;

            // For testing: show the number of steps. This helps to establish whether the rays are correctly oriented
            // gl_FragColor = vec4(0.0, float(nsteps) / size.x, 1.0, 1.0);
            // return;

            bool sliceHit = false;
            bool segmentHit = false;
            bool labelHit = false;
            vec3 p; vec3 pn; vec3 pf;

            if (sliceVisible) {
              sliceXYZ(p, sliceHit, slice, sdfRayOrigin, sdfRayDirection);

              if (sliceHit) {
                float v = texture(volumeTex, p.xyz + vec3( 0.5 )).r;
                gl_FragColor = vec4(v, v, v, 1.0);
              }
            }

            if (segmentVisible) {
              vec4 boxNearPoint = vec4( sdfRayOrigin + sdfRayDirection * ( distToBox + 1e-5 ), 1.0 );
              vec4 boxFarPoint = vec4( sdfRayOrigin + sdfRayDirection * ( distToBox + distInsideBox - 1e-5 ), 1.0 );
              vec3 pn = (sdfTransform * boxNearPoint).xyz;
              vec3 pf = (sdfTransform * boxFarPoint).xyz;
              
              // near -> surface
              segment(pn, segmentHit, surface, rayDirection);

              if (segmentHit) {
                p = (sdfTransform * vec4(p, 1.0)).xyz;
                if (sliceHit && length(p - rayOrigin) < length(pn - rayOrigin)) return;

                // far -> surface
                segment(pf, segmentHit, surface, -rayDirection);

                vec3 uv = (sdfTransformInverse * vec4(pn, 1.0)).xyz + vec3( 0.5 );
                vec4 volumeColor;
                
                if (colorful) {
                  // volume
                  float thickness = length(pf - pn);
                  nsteps = int(thickness * size.x / relative_step_size + 0.5);
                  if ( nsteps < 1 ) discard;
                  vec3 step = sdfRayDirection * thickness / float(nsteps);
                  volumeColor = cast_mip(uv, step, nsteps, sdfRayDirection);

                  // label
                  segment(pn, labelHit, tlabel, rayDirection);
                  if (labelHit) {
                    float v = texture(labelTex, uv).r;
                    vec4 labelColor = vec4(v, v, v, 1.0);
                    gl_FragColor = label * labelColor + (1.0 - label) * volumeColor;
                    return;
                  }
                } else {
                  // volume
                  float v = texture(volumeTex, uv).r;
                  volumeColor = vec4(v, v, v, 1.0);

                  // label
                  float d = texture(sdfTex, uv).r;
                  if (d < tlabel) {
                    float ink = texture(labelTex, uv).r;
                    vec4 labelColor = vec4(ink, ink, ink, 1.0);
                    gl_FragColor = label * labelColor + (1.0 - label) * volumeColor;
                    return;
                  }
                }

                gl_FragColor = volumeColor; return;
              }
            }
          }

          if (gl_FragColor.a < 0.05){ discard; }
        }

        // SDF ray march
        void segment(inout vec3 p, inout bool hit, float surface, vec3 rayDirection) {
          for ( int i = 0; i < MAX_STEPS; i ++ ) {
            // sdf box extends from - 0.5 to 0.5
            // transform into the local bounds space [ 0, 1 ] and check if we're inside the bounds
            vec3 uv = ( sdfTransformInverse * vec4(p, 1.0) ).xyz + vec3( 0.5 );
            // get the distance to surface and exit the loop if we're close to the surface
            float distanceToSurface = texture( sdfTex, uv ).r - surface;
            if ( distanceToSurface < SURFACE_EPSILON ) { hit = true; break; }
            if ( uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || uv.z < 0.0 || uv.z > 1.0 ) { break; }
            // step the ray
            p += rayDirection * abs( distanceToSurface );
          }
        }

        vec4 apply_colormap(float val) {
          float v = (val - clim[0]) / (clim[1] - clim[0]);
          return texture2D(cmdata, vec2(v, 0.5));
          // return vec4(vec3(val), 1.0);
        }

        vec4 cast_mip(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray) {
          float max_val = -1e6;
          int max_i = 100;
          vec3 loc = start_loc;

          // float val = texture(volumeTex, start_loc).r;
          // return apply_colormap(val);

          // Enter the raycasting loop. In WebGL 1 the loop index cannot be compared with
          // non-constant expression. So we use a hard-coded max, and an additional condition
          // inside the loop.
          for (int iter=0; iter<MAX_STEPS; iter++) {
            if (iter >= nsteps)
              break;
            // Sample from the 3D texture
            float val = texture(volumeTex, loc).r;
            // Apply MIP operation
            if (val > max_val) {
              max_val = val;
              max_i = iter;
            }
            // Advance location deeper into the volume
            loc += step;
          }

          // Resolve final color
          return apply_colormap(max_val);
        }
      `,
    })
  }
}
