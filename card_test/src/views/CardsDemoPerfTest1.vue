<template>
  <div class="demo-container">
    <div class="controllers-container">
      <p><span>card count</span>
      <input type="range" min="1" :max="maxCardCount" v-model="cardCount" />
      </p>
      <p>
        <span>Framerate: {{fps}} fps</span>
      </p>
    </div>
    <card-dynamic-container
    :hasDampMotion="false">
      <inner-dynamic-container>
        <dynamic-card-follow-mouse v-for="(item, i) in visibleCardContents" :key="i"
        :translate="item.translate"
        :rotate="item.rotate"
        :blur="item.blur"
        :cardSize="[cardSize, cardSize]"
        :mousePosX="mousePosX"
        :mousePosY="mousePosY"
        :screenResized="screenResized"
        class="card__flex-vertical"
        >
          <template v-slot:front>
            <span class="card-front-style">{{item.text}}</span>
          </template>
        </dynamic-card-follow-mouse>
      </inner-dynamic-container>
    </card-dynamic-container>
  </div>
</template>

<script>
import DynamicCardFollowMouse from '@/components/DynamicCardFollowMouse'
export default {
  components: {
    DynamicCardFollowMouse
  },
  data() {
    return {
      /**
       * in cardContents,
       *    translate indicates each card's translation **relative to table center**
       */
      cardContents: [],
      cardTotalSize: 800,
      cardSize: 80,
      cardCount: 1,
      maxCardCount: 12,
      mousePosX: 0,
      mousePosY: 0,
      screenResized: 0,

      // fps
      frame: 0,
      allFrameCount: 0,
      lastTime: Date.now(),
      lastFameTime: Date.now(),
      fps: 60,
      stop: undefined
    }
  },
  computed: {
    visibleCardContents() {
      return this.cardContents.slice(0, this.cardCount * this.cardCount)
    }
  },
  created() {
    let globalAff = (this.maxCardCount - 1) * this.cardSize / 2
    this.cardSize = this.cardTotalSize / this.maxCardCount
    for (let i = 0; i < this.maxCardCount * this.maxCardCount; i++) {
      this.cardContents.push(
        {
          translate: [Math.floor(i/this.maxCardCount)*this.cardSize - globalAff,(i%this.maxCardCount)*this.cardSize - globalAff,0],
          rotate: [0,0,0],
          blur: 0,
          text: "card" + (i+1)
        }
      )
    }
    // this.cardMove__Lay()
    // this.swipeInterval = setInterval(this.updateLeftmostPosition, "50")
    // window.addEventListener('mousemove', this.getMousePosX)
    // this.maxLeftmostPosition = (this.cardContents.length - 1) * this.cardLayGap
  },
  mounted() {
    window.addEventListener("mousemove", this.cardFaceMouse)
    window.addEventListener("resize", this.screenResize)
    this.stop = window.requestAnimationFrame(() => {this.loopOnce()})
    this.cardCount = 2
  },
  methods: {
    cardCountChanged() {
      this.screenResize()
      let count = this.cardCount
      this.cardSize = this.cardTotalSize / this.cardCount
      let globalAff = (count - 1) * this.cardSize / 2
      for (let i = 0; i < count*count; i++) {
        this.cardContents[i].translate = [Math.floor(i/count)*this.cardSize - globalAff,(i%count)*this.cardSize - globalAff,0]
      }
    },
    cardFaceMouse(e) {
      this.mousePosX = e.clientX
      this.mousePosY = e.clientY
    },
    screenResize() {
      this.screenResized = (this.screenResized + 1) % 2
    },
    loopOnce() {
      const self = this
      var now = Date.now();
  
      self.lastFameTime = now;
      // 不置 0，在动画的开头及结尾记录此值的差值算出 FPS
      self.allFrameCount++;
      self.frame++;
  
      if (now > 1000 + self.lastTime) {
        var t_fps = Math.round((self.frame * 1000) / (now - self.lastTime));
        self.fps = t_fps
        self.frame = 0;
        self.lastTime = now;
      }
      self.stop = window.requestAnimationFrame(() => {this.loopOnce()})
    }
  },
  beforeUnmount() {
    // this.cardMove__Lay()
    // clearInterval(this.updateLeftmostPosition)
    // window.removeEventListener('mousemove', this.getMousePosX)
    window.removeEventListener("mousemove", this.cardFaceMouse)
    window.removeEventListener("resize", this.screenResize)

    const r_rAF = function () {
    return (
            window.cancelAnimationFrame ||
            window.webkitCancelAnimationFrame
        );
    }();
    r_rAF(this.stop)
  },
  watch: {
    cardCount() {
      this.cardCountChanged()
    }
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
.controllers-container {
  z-index: 10;
  position: absolute;
  top: 100px;
  background: #fff;
  padding: 10px;
}
.controllers-container span {
  font-size: 20pt;
  font-weight: bold;
  color: red;
}
.card__flex-vertical >>> .card__front {
  flex-direction: column;
}
.controllers-container button {
  height: 50pt;
  width: 100pt;
  background: #f00;
}
</style>