<template>
  <!-- <div class="card-button-group">
    <button @click="cardArrangement = 0">stack</button>
    <button @click="cardArrangement = 1">carousel</button>
  </div> -->
  <div class="card-container" :style="{
    transform: `translateX(${initialTranslate[0]}pt) translateY(${initialTranslate[1]}pt) translateZ(${initialTranslate[2]}pt)
    rotateX(${initialRotate[0]+dampRotY}deg) rotateY(${initialRotate[1]-dampRotX}deg)`
  }"
  >
  <slot></slot>
  </div>
</template>

<script>
import { IsMobile } from "@/js/Utils"
export default {
  name: "CardDynamicContainer",
  props: {
    initialTranslate: {
      type: Array,
      default: () => [0,0,0],
    },
    initialRotate: {
      type: Array,
      default: () => [0,0,0],
    },
    hasDampMotion: {
      type: Boolean,
      default: true,
    },
  },
  data() {
    return {
      // mouse damp
      bufferDamp: 0.9,
      mousePosX: -1,
      mousePosY: -1,
      dampPosX: -1,
      dampPosY: -1,
      dampRotX: 0,
      dampRotY: 0,
      dampMinValue: 1e-3,
      dampRotIntensity: 0.02,
      dampInterval: undefined,

      // touch
      touchBeginX: 0,
      touchBeginY: 0,
      touchPosX: 0,
      touchPosY: 0
    }
  },
  created() {
    if (this.hasDampMotion) {
      this.dampInterval = setInterval(this.updateMouseDamp, "50")
      window.addEventListener('mousemove', this.getMousePos)
  
      // calculate carousel wheel radius = card_wid/(2*atan(pi/n))
      // this.carouselWheelRadius = 180/(2*Math.atan(Math.PI/this.cardContents.length))
      
      window.addEventListener("wheel", this.mouseWheel)
      if (IsMobile()) {
        window.addEventListener('touchstart',this.touchs,false)
        window.addEventListener('touchmove',this.touchs,false)
        window.addEventListener('touchend',this.touchs,false)
      }
      else {
        window.addEventListener('touchstart',this.touchs,false)
        window.addEventListener('touchmove',this.touchs,false)
        window.addEventListener('touchend',this.touchs,false)
      }
    }
  },
  methods: {
    touchs(event) {
      let touch = event.touches[0]
      if (event.type == 'touchmove') {
        this.touchPosX = touch.clientX
        this.touchPosY = touch.clientY
      }
      else if (event.type == 'touchstart') {
        this.touchBeginX = touch.clientX
        this.touchBeginY = touch.clientY
        this.touchPosX = touch.clientX
        this.touchPosY = touch.clientY
      }
      else if (event.type == 'touchend' || event.type == 'touchcancel') {
        console.log('asd')
        let touchMoveDistX = this.touchPosX - this.touchBeginX
        let touchMoveDistY = this.touchPosY - this.touchBeginY
        if (Math.abs(touchMoveDistY) < Math.abs(touchMoveDistX) * 0.5 && Math.abs(touchMoveDistX) > 50) {
          if (this.cardArrangement !== 1) {
            return
          }
          this.mouseWheel({deltaY: touchMoveDistX})
          // if (touchMoveDistX > 0) {
          //   // swipe right
          // }
          // else {
          //   // swipe left
          // }
        }
      }
      // console.log(touch, event.type)
    },
    getMousePos(val) {
      this.mousePosX = val.clientX
      this.mousePosY = val.clientY
      if (this.dampInterval == undefined) {
        this.dampInterval = setInterval(this.updateMouseDamp, "20")
      }
    },
    updateMouseDamp() {
      if (this.dampPosX != -1 || this.dampPosY != -1) {
        this.dampPosX = this.bufferDamp * this.dampPosX + (1-this.bufferDamp) * this.mousePosX
        this.dampPosY = this.bufferDamp * this.dampPosY + (1-this.bufferDamp) * this.mousePosY
        this.dampRotX = this.mousePosX - this.dampPosX
        this.dampRotY = this.mousePosY - this.dampPosY
        this.dampRotX *= this.dampRotIntensity
        this.dampRotY *= this.dampRotIntensity
        if (Math.abs(this.dampRotX) < this.dampMinValue && Math.abs(this.dampRotY) < this.dampMinValue) {
          this.dampRotX = 0
          this.dampRotY = 0
          clearInterval(this.dampInterval)
          this.dampInterval = undefined
        }
      }
      else {
        this.dampPosX = this.mousePosX
        this.dampPosY = this.mousePosX
      }
    }
  },
  beforeUnmount () {
    if (this.hasDampMotion) {
      clearInterval(this.dampInterval)
      window.removeEventListener('mousemove', this.getMousePos)
  
      // calculate carousel wheel radius = card_wid/(2*atan(pi/n))
      // this.carouselWheelRadius = 180/(2*Math.atan(Math.PI/this.cardContents.length))
      
      window.removeEventListener("wheel", this.mouseWheel)
      if (IsMobile()) {
        window.removeEventListener('touchstart',this.touchs,false)
        window.removeEventListener('touchmove',this.touchs,false)
        window.removeEventListener('touchend',this.touchs,false)
      }
      else {
        window.removeEventListener('touchstart',this.touchs,false)
        window.removeEventListener('touchmove',this.touchs,false)
        window.removeEventListener('touchend',this.touchs,false)
      }
    }
  }
}
</script>

<style scoped>
.card-container {
  position: relative;
  height: 0;
  width: 0;
  perspective: 1000px;
  transform-style: preserve-3d;
  transform-origin: 0 0 0;
  display: flex;
  justify-content: center;
  align-items: center;
  /* display: flex;
  justify-content: center;
  align-items: center; */
}
</style>