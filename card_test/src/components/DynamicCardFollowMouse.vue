<template>
  <div class="card__basic-style transition__none"
  ref="card"
  :class="{transition__basic: doTransition, 'is-front': isFront}"
  :style="{
    width: cardSize[0]+'pt',
    height: cardSize[1]+'pt',
    transform: cardTransform
  }"
  @click="$emit('clicked', $event)"
  >
    <div class="card__front">
      <div style="display: flex">
        <button class="front-btn" @click="bringToFront">f</button>
        <button class="back-btn" @click="bringToBack">b</button>
      </div>
      <slot name="front">card</slot>
    </div>
    <div class="card__back">
      <slot name="back">card</slot>
    </div>
    <slot name="icing"></slot>
  </div>
</template>

<script>
export default {
  name: "DynamicCardFollowMouse",
  props: {
    cardSize: {
      type: Array,
      default: () => [10,10],
    },
    translate: {
      type: Array,
      default: () => [0,0,0],
    },
    rotate: {
      type: Array,
      default: () => [0,0,0],
    },
    blur: {
      type: Number,
      default: 0,
    },
    doTransition: {
      type: Boolean,
      default: true,
    },
    mousePosX: {
      type: Number,
      default: 0,
    },
    mousePosY: {
      type: Number,
      default: 0,
    },
    screenResized: {
      type: Number,
      default: 0,
    },
  },
  data() {
    return {
      isFront: false,
      card: undefined,
      internalRotateIntensity: 0.05,
      rotX: 0,
      rotY: 0,
      bcr: [0, 0]
    }
  },
  mounted() {
    this.card = this.$refs['card']
    this.bcr = this.card.getBoundingClientRect()
  },
  computed: {
    mousePosMove() {
      return (this.mousePosX+1) / (this.mousePosY+3)
    },
    cardTransform() {
      if (this.isFront) {
        return `translateZ(${200}pt)`
      }
      else {
        try {
          return `translateX(${this.translate[0]}pt)
            translateY(${this.translate[1]}pt)
            translateZ(${this.translate[2]}pt)
            rotateX(${this.rotate[0]+this.rotY}deg)
            rotateY(${this.rotate[1]-this.rotX}deg)
            rotateZ(${this.rotate[2]}deg)`
        } catch {
          return ''
        }
      }
    }
  },
  methods: {
    bringToFront() {
      this.isFront = true
      this.$emit('front', this.card)
      // this.translate = [0,0,0]
      // this.rotate = [0,0,0]
    },
    bringToBack() {
      this.isFront = false
      this.$emit('back', this.card)
      // this.translate = [0,0,0]
      // this.rotate = [0,0,0]
    }
  },
  watch: {
    screenResized() {
      this.$nextTick(() => {
        this.bcr = this.card.getBoundingClientRect()
      })
    },
    mousePosMove() {
      this.rotX = Math.round((this.bcr.x + this.cardSize[0]/2 - this.mousePosX) * this.internalRotateIntensity)
      this.rotY = Math.round((this.bcr.y + this.cardSize[1]/2 - this.mousePosY) * this.internalRotateIntensity)
    }
  }
}
</script>

<style scoped>
.card__back {
  background-image: url('https://opengameart.org/sites/default/files/card%20back%20black.png');
  background-size: 100% 100%;
}
.front-btn, .back-btn {
  background: rgb(172, 0, 0);
  height: 16pt;
  width: 16pt;
  border: none;
  border-radius: 100%;
  color: #fff;
}
.back-btn {
  background: rgb(0, 114, 0);
}
</style>