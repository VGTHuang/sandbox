<template>
  <div class="card-container_no-3d">
    <div class="info-container">
      <p>
        url: {{p1}} / {{p2}}
      </p>
      <div>
        <el-radio v-model="switchStyle" :label="1" size="large">Switch page: arrows</el-radio>
        <el-radio v-model="switchStyle" :label="2" size="large">Switch page: steppers</el-radio>
      </div>
    </div>
    <div v-if="switchStyle==1">
      <div class="arrow-left-container">
        <el-button circle size="large" :disabled="p2 <= 1"
        @click="leftClick"
        >
          <el-icon><ArrowLeftBold /></el-icon>
        </el-button>
      </div>
      <div class="arrow-right-container">
        <el-button circle size="large" :disabled="p2 >= maxP2"
        @click="rightClick"
        >
          <el-icon><ArrowRightBold /></el-icon>
        </el-button>
      </div>
    </div>
    <div v-else>
      <div class="stepper-left-container">
        <div
        class="stepper-indicator"
        :class="{'stepper-indicator__selected': p1 == i}"
        v-for="i in p1Indicators" :key="i"
        @click="forceSwitchP1(i)"
        ></div>
      </div>
      <div class="stepper-bottom-container">
        <div
        class="stepper-indicator"
        :class="{'stepper-indicator__selected': p2 == i}"
        v-for="i in p2Indicators" :key="i"
        @click="forceSwitchP2(i)"
        ></div>
      </div>
    </div>
    <card-dynamic-container
    :hasDampMotion="false">
      <inner-dynamic-container>
        <dynamic-card v-for="(item, i) in visibleCardContents" :key="i"
        :translate="item.translate"
        :rotate="item.rotate"
        :cardSize="[cardSize, cardSize]"
        :noFrontBackButton="true"
        class="transition__slow"
        >
          <template v-slot:front>
            <el-skeleton :loading="item.text == undefined" animated style="padding: 10px" >
              <template #template>
                  <el-skeleton-item variant="h3" style="width: 50%; margin-bottom: 10pt" />
                  <el-skeleton-item variant="text" :rows="3" />
              </template>
              <template #default>
                <el-card :body-style="{ padding: '0px', marginBottom: '1px' }">
                  <div>{{item.text}}</div>
                </el-card>
              </template>
            </el-skeleton>
          </template>
          <template v-slot:back>
            <el-skeleton :loading="item.text == undefined" animated style="padding: 10px" >
              <template #template>
                  <el-skeleton-item variant="h3" style="width: 50%; margin-bottom: 10pt" />
                  <el-skeleton-item variant="text" :rows="3" />
              </template>
              <template #default>
                <el-card :body-style="{ padding: '0px', marginBottom: '1px' }">
                  <div>{{item.text}}</div>
                </el-card>
              </template>
            </el-skeleton>
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
      cardContents: [],
      cardCount: 3,
      totalCardCount: 9,
      cardSize: 100,
      cardTotalSize: 400,
      
      maxP1: 3,
      maxP2: 3,

      p1Indicators: [],
      p2Indicators: [],
      switchStyle: 1,
    }
  },
  computed: {
    p1() {
      return parseInt(this.$route.params.p1);
    },
    p2() {
      return parseInt(this.$route.params.p2);
    },
    visibleCardContents() {
      return this.cardContents.slice(0, this.cardCount)
    }
  },
  created() {
    this.p1Indicators = Array.from(Array(this.maxP1), (v,k) =>k+1)
    this.p2Indicators = Array.from(Array(this.maxP2), (v,k) =>k+1)
    this.cardSize = this.cardTotalSize / this.cardCount
    for (let i = 0; i < this.totalCardCount; i++) {
      this.cardContents.push(
        {
          translate: [this.cardSize * (i - this.totalCardCount/2), 0, 0],
          rotate: [20,0,0],
          blur: 0,
          text: "card" + (i+1)
        }
      )
    }
  },
  mounted() {
    window.addEventListener('mousewheel', this.mousewheelSwitchP1)
    this.changeCardsByUrl()
  },
  methods: {
    mousewheelSwitchP1(val) {
      if (val.deltaY > 0) {
        // scroll down
        let nextP1 = this.p1 + 1
        if (nextP1 <= this.maxP1) {
          this.$router.push(`/cards_demo_pages/${nextP1}/1`)
        }
      } else {
        let nextP1 = this.p1 - 1
        if (nextP1 >= 1) {
          this.$router.push(`/cards_demo_pages/${nextP1}/1`)
        }
      }
    },
    forceSwitchP1(p1Val) {
      if (p1Val <= this.maxP1 && p1Val >= 1) {
        this.$router.push(`/cards_demo_pages/${p1Val}/1`)
      }
    },
    forceSwitchP2(p2Val) {
      if (p2Val <= this.maxP2 && p2Val >= 1) {
        this.$router.push(`/cards_demo_pages/${this.p1}/${p2Val}`)
      }
    },
    leftClick() {
      let nextP2 = this.p2 - 1
      if (nextP2 >= 1) {
        this.$router.push(`/cards_demo_pages/${this.p1}/${nextP2}`)
      }
    },
    rightClick() {
      let nextP2 = this.p2 + 1
      if (nextP2 <= this.maxP2) {
        this.$router.push(`/cards_demo_pages/${this.p1}/${nextP2}`)
      }
    },
    setCardCententText(cardContent, text) {
      cardContent.text = undefined
      setTimeout(() => {
        cardContent.text = text
      }, 1000)
    },
    changeCardsByUrl() {
      const cardTextPref = `Section ${this.p1}, Sub-section${this.p2}`
      switch(this.p2) {
        case 1:
          this.cardCount = 3
          this.cardSize = 300
          for (let i = 0; i < 3; i++) {
            this.cardContents[i].translate[0] = (i - 1) * this.cardSize
            this.cardContents[i].translate[1] = (i - 1) * 20
            this.cardContents[i].translate[2] = 0
            this.cardContents[i].rotate[0] = (i+1)*5
            // this.cardContents[i].rotate[1] = -(i-1)*5
            this.cardContents[i].rotate[2] = 0
            this.setCardCententText(this.cardContents[i], cardTextPref)
          }
          break
        case 2:
          this.cardCount = 6
          this.cardSize = 150
          for (let i = 0; i < 3; i++) {
            this.cardContents[i].translate[0] = (i - 1) * this.cardSize
            this.cardContents[i].translate[1] = -this.cardSize / 2
            this.cardContents[i].translate[2] = 0
            this.cardContents[i].rotate[0] = 20
            // this.cardContents[i].rotate[1] = 20 * (1-i)
            this.cardContents[i].rotate[2] = 0
            this.setCardCententText(this.cardContents[i], cardTextPref)
          }
          for (let i = 3; i < 6; i++) {
            this.cardContents[i].translate[0] = (i - 4) * this.cardSize
            this.cardContents[i].translate[1] = this.cardSize / 2
            this.cardContents[i].translate[2] = 0
            this.cardContents[i].rotate[0] = 20
            // this.cardContents[i].rotate[1] = 20 * (4-i)
            this.cardContents[i].rotate[2] = 0
            this.setCardCententText(this.cardContents[i], cardTextPref)
          }
          break
        case 3:
          this.cardCount = 9
          this.cardSize = 100
          for (let i = 0; i < 9; i++) {
            this.cardContents[i].translate[0] = (i%3 - 1) * this.cardSize
            this.cardContents[i].translate[1] = (Math.floor(i/3) - 1) * this.cardSize
            this.cardContents[i].translate[2] = 0
            this.cardContents[i].rotate[0] = 20
            // this.cardContents[i].rotate[1] = 20
            this.cardContents[i].rotate[2] = 0
            this.setCardCententText(this.cardContents[i], cardTextPref)
          }
          break
      }
    },
  },
  beforeUnmount() {
    window.removeEventListener('mousewheel', this.mousewheelSwitchP1)
  },
  watch: {
    p1(oldVal, newVal) {
      console.log(oldVal, newVal)
      this.changeCardsByUrl()
      if (oldVal < newVal) {
        for (let i = 0; i < this.totalCardCount; i++) {
          this.cardContents[i].rotate[1] += 180
        }
      }
      else {
        for (let i = 0; i < this.totalCardCount; i++) {
          this.cardContents[i].rotate[1] -= 180
        }
      }
    },
    p2(oldVal, newVal) {
      console.log(oldVal, newVal)
      this.changeCardsByUrl()
    }
  }
}
</script>

<style scoped>
.card-container_no-3d {
  position: relative;
  height: 0;
  width: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  /* display: flex;
  justify-content: center;
  align-items: center; */
}
.page-container {
  background: #ddd;
}
.arrow-left-container, 
.arrow-right-container {
  height: 100%;
  width: 50pt;
  top: 0;
  /* background: #aaa; */
  position: absolute;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 100;
}
.arrow-left-container {
  left: 0;
}
.arrow-right-container {
  right: 0;
}
.stepper-left-container,
.stepper-bottom-container {
  height: 50pt;
  width: 50pt;
  position: absolute;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 100;
}
.stepper-left-container {
  height: 100%;
  width: 50pt;
  left: 0;
  top: 0;
  flex-direction: column;
}
.stepper-bottom-container {
  height: 50pt;
  width: 100%;
  left: 0;
  bottom: 0;
}
.info-container {
  position: absolute;
  top: 100px;
  background: #fff;
  padding: 10px;
  font-size: 20pt;
  font-weight: bold;
  color: red;
}
.info-container >>> .el-radio__label {
  font-size: 16pt;
  font-weight: bold;
  color: red;
}
.stepper-indicator {
  height: 20pt;
  width: 20pt;
  border-radius: 100%;
  margin: 10pt;
  background: #ccc;
  transition: 250ms;
}
.stepper-indicator__selected {
  background: #f00;
}
</style>