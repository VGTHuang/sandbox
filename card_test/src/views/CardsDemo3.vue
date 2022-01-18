<template>
  <div class="demo-container"
    @click="containerClicked">
    <card-dynamic-container
    @containerClicked="containerClicked">
      <inner-dynamic-container
      :translate="containerTranslate">
        <dynamic-card v-for="(item, i) in cardContents" :key="i"
        :translate="item.translate"
        :rotate="item.rotate"
        :blur="item.blur"
        @front="cardBroughtToFront($event, i)"
        class="demo3-card"
        >
          <template v-slot:front>
            <span class="card-front-style">{{item.text}}</span>
          </template>
          <template v-slot:icing>
            <div class="demo3-card-icing-container demo3-card-icing-container__left transition__basic" v-if="i==0">1</div>
            <div class="demo3-card-icing-container demo3-card-icing-container__right transition__basic transition__slow" v-if="i==0">2</div>
            <div class="demo3-card-icing-funky" v-if="i==1"></div>
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
      cardLayGap: 200,
      containerTranslate: [-100,0,0]
    }
  },
  created() {
    for (let i = 0; i < 2; i++) {
      this.cardContents.push(
        {
          translate: [this.cardLayGap * i,0,0],
          rotate: [45,0,0],
          blur: 0,
          text: "card" + (i+1)
        }
      )
    }
  },
  methods: {
    cardBroughtToFront(e, val) {
      this.$nextTick(() => {
        e.style.transform = `translateZ(${200}pt) translateX(${val*this.cardLayGap}pt) rotateX(10deg)`
      })
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
.card-front-style {
  font-size: 40pt;
}
.demo3-card {
  background: #cccd;
}
.demo3-card-icing-container {
  height: 40pt;
  width: 80pt;
  position: absolute;
  top: 0pt;
  transform: translateZ(-50pt);
  font-size: 20pt;
  opacity: 0;
  background: #ccc;
}
.demo3-card-icing-container__left {
  left: 0pt;
}
.demo3-card-icing-container__right {
  right: 0pt;
}
.is-front >>> .demo3-card-icing-container {
  top: -80pt;
  opacity: 1;
  /* transition: ease; */
}
.demo3-card-icing-funky {
  height: 80pt;
  width: 80pt;
  position: absolute;
  top: 0;
  right: 0;
  background: url(/kh.png);
  background-size: 80pt 80pt;
  opacity: 0;
  transform: translateZ(-20pt);
  transform-origin: 0 80pt 0;
}
@keyframes example {
  0%   {transform: translateY(-20pt) translateZ(-20pt) rotateZ(0deg)}
  60%  {transform: translateY(-100pt) translateZ(-20pt) rotateZ(-30deg)}
  100% {transform: translateY(-90pt) translateZ(-20pt) rotateZ(0deg)}
}

.is-front >>> .demo3-card-icing-funky {
  opacity: 1;
  transform: translateY(-90pt) translateZ(-20pt) rotateZ(0deg);
  animation-name: example;
  animation-duration: 500ms;
  /* transition: ease; */
}
</style>