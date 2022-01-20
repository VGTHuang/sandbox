<template>
  <div class="demo-container"
  :class="{'demo-container-w-bg': bg>0}">
    <card-dynamic-container
    :hasDampMotion="false"
    @containerClicked="containerClicked">
      <inner-dynamic-container
      :translate="containerTranslate">
        <dynamic-card v-for="(item, i) in cardContents" :key="i"
        :translate="item.translate"
        :rotate="item.rotate"
        :blur="item.blur"
        :noFrontBackButton="true"
        :noDefaultBack="true"
        @mouseenter="cardMouseEnter($event, i)"
        @mouseleave="cardMouseLeave($event, i)"
        @front="cardBroughtToFront($event, i)"
        @back="cardBroughtToBack($event, i)"
        class="lay-card"
        >
          <template v-slot:front>
            <button @click="cardFlip($event, i, 1)">flip</button>
            <el-image style="width: 180px; height: 180px" :src="item.frontUrl">
              <template #placeholder>
                <el-skeleton animated>
                  <template #template>
                    <el-skeleton-item variant="image" animated style="width: 180px; height: 180px" />
                  </template>
                </el-skeleton>
              </template>
              <template #error>
                <div class="image-slot">
                  <el-icon><Picture /></el-icon>
                </div>
              </template>
              </el-image>
          </template>
          <template v-slot:back>
            <button @click="cardFlip($event, i, 2)">flip</button>
            <el-image style="width: 180px; height: 180px" :src="item.backUrl">
              <template #placeholder>
                <el-skeleton animated>
                  <template #template>
                    <el-skeleton-item variant="image" animated style="width: 180px; height: 180px" />
                  </template>
                </el-skeleton>
              </template>
              <template #error>
                <div class="image-slot">
                  <el-icon><Picture /></el-icon>
                </div>
              </template>
            </el-image>
          </template>
        </dynamic-card>
      </inner-dynamic-container>
    </card-dynamic-container>
  </div>
</template>

<script>
export default {
  data() {
    return {
      /**
       * in cardContents,
       *    translate indicates each card's translation **relative to table center**
       */
      cardContents: [],

      // rand image
      randImageId: 1,

      // lay
      cardLayGap: 200,
      swipeInterval: undefined,
      leftmostPosition: 0,
      maxLeftmostPosition: 300,
      layRotateAngle: 45,

      // swipe
      mousePosX: -1,
      mousePosY: -1,
      swipeSpeed: 2000,
      swipeDeadzoneWidth: 0.2,

      containerTranslate: [0,0,0],
      stopMotion: false
    }
  },
  created() {
    for (let i = 0; i < 7; i++) {
      this.cardContents.push(
        {
          translate: [0,0,0],
          rotate: [0,0,0],
          blur: 0,
          frontUrl: this.getRandImage(),
          backUrl: this.getRandImage()
        }
      )
    }
    this.cardMove__Lay()
    this.swipeInterval = setInterval(this.updateLeftmostPosition, "50")
    window.addEventListener('mousemove', this.getMousePosX)
    this.maxLeftmostPosition = (this.cardContents.length - 1) * this.cardLayGap
  },
  computed: {
    bg() {
      return this.$route.params.bg;
    }
  },
  methods: {
    getRandImage() {
      this.randImageId = Math.floor(Math.random() * 1024)
      return `https://picsum.photos/id/${this.randImageId}/180/180`
    },
    cardMove__Lay() {
      this.containerTranslate[0] = this.leftmostPosition
      for(let i = 0; i < this.cardContents.length; i++) {
        let tempContent = this.cardContents[i]
        let pos = i * this.cardLayGap
        tempContent.rotate = [this.layRotateAngle,0,0]
        tempContent.translate = [pos, 0, 0]
        tempContent.blur = 0
      }
    },
    getMousePosX(val) {
      this.mousePosX = val.clientX / document.body.clientWidth
      this.mousePosY = val.clientY
    },
    updateLeftmostPosition() {
      let relativeSpeed = Math.abs(0.5 - this.mousePosX) - this.swipeDeadzoneWidth / 2
      // console.log(relativeSpeed)
      if (relativeSpeed < 0) {
        return
      }
      if (this.leftmostPosition >= 0 && this.mousePosX < 0.5) {
        this.leftmostPosition = 0
      }
      else if (this.leftmostPosition <= -this.maxLeftmostPosition && this.mousePosX > 0.5) {
        this.leftmostPosition = -this.maxLeftmostPosition
      }
      else {
        relativeSpeed = Math.pow(relativeSpeed, 4)
        if (this.mousePosX < 0.5) {
          this.leftmostPosition += this.swipeSpeed * relativeSpeed
        }
        else {
          this.leftmostPosition -= this.swipeSpeed * relativeSpeed
        }
      }
      this.updateCardRowPositions()
    },
    updateCardRowPositions(forced) {
      if (!this.stopMotion || forced) {
        this.containerTranslate[0] = this.leftmostPosition
      }
      // for(let i = 0; i < this.cardContents.length; i++) {
      //   let tempContent = this.cardContents[i]
      //   let pos = this.leftmostPosition + i * this.cardLayGap
      //   // tempContent.rotate = [45,0,0]
      //   tempContent.translate = [pos, 0, 0]
      //   // tempContent.blur = 0
      // }
    },
    cardMouseEnter(e, val) {
      this.cardContents[val].rotate[0] = 20
    },
    cardMouseLeave(e, val) {
      this.cardContents[val].rotate[0] = this.layRotateAngle
    },
    // cardBroughtToFront(e, val) {
    //   this.stopMotion = true
    //   this.leftmostPosition = -this.cardLayGap * val
    //   this.$nextTick(() => {
    //     e.style.transform = `translateZ(${200}pt) translateX(${-this.leftmostPosition}pt)`
    //   })
    //   this.updateCardRowPositions(true)
    // },
    // cardBroughtToBack(e, val) {
    //   this.stopMotion = false
    //   this.leftmostPosition = -this.cardLayGap * val
    //   this.updateCardRowPositions()
    // },
    cardFlip(e, val, i) {
      console.log(this.cardContents[val].rotate)
      this.cardContents[val].rotate[1] += 180
      if (i == 1) {
        this.cardContents[val].backUrl = this.getRandImage()
      }
      else {
        this.cardContents[val].frontUrl = this.getRandImage()
      }
    }
  },
  watch: {
    bg(val) {
      console.log(val)
    }
  },
  beforeUnmount() {
    // this.cardMove__Lay()
    clearInterval(this.updateLeftmostPosition)
    window.removeEventListener('mousemove', this.getMousePosX)
  }
}
</script>

<style scoped>
.demo-container {
  height: 100%;
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
}
.lay-card {
  transform-origin: 50% 100% 0;
}
.lay-card >>> .card__front, 
.lay-card >>> .card__back {
  flex-direction: column;
}
</style>