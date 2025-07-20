import { Component, OnDestroy, OnInit, AfterViewInit } from '@angular/core';
import { FirebaseApp, initializeApp } from "firebase/app";
import * as tf from '@tensorflow/tfjs';
import { Database, get, getDatabase, onValue, ref } from 'firebase/database';

interface TrafficLight {
  id: number;
  activeLight: 'red' | 'yellow' | 'green';
  position: { x: number, y: number };
}
interface PedestrianSign {
  id: number;
  activeLight: 'walk' | 'dontWalk';
  position: { x: number, y: number };
}
interface Car {
  id: number;
  position: { x: number, y: number };
  initialPosition: { x: number, y: number };
  totalNumCars: number;
}
interface Pedestrian {
  id: number;
  position: { x: number, y: number };
  initialPosition: { x: number, y: number };
  totalNumPeds: number;
}

type TrafficState = 'base' | 'car_only' | 'pedestrian'; // --- State machine definition ---

@Component({
  selector: 'app-stop-light',
  templateUrl: './stop-light.component.html',
  styleUrls: ['./stop-light.component.scss']
})
export class StopLightComponent implements OnInit, AfterViewInit, OnDestroy{
  trafficLights: TrafficLight[] = [
      { id: 0, activeLight: 'red', position: { x: 950, y: 50 } },  //north
      { id: 1, activeLight: 'red', position: { x: 600, y: 200 } }, //west
      { id: 2, activeLight: 'red', position: { x: 1300, y: 500 } } //east
  ];
  pedestrianSigns: PedestrianSign[] = [
      { id: 0, activeLight: 'dontWalk', position: { x: 740, y: 400 } }, //west
      { id: 1, activeLight: 'dontWalk', position: { x: 1125, y: 400 } },//east
      { id: 2, activeLight: 'dontWalk', position: { x: 950, y: 660 } }  //south
  ];
  cars: Car[] = [
      { id: 0, position: { x: 950, y: 825 }, initialPosition: { x: 950, y: 825 }, totalNumCars: 0 },  //north
      { id: 1, position: { x: 1300, y: 250 }, initialPosition: {x: 1300, y: 250 }, totalNumCars: 0 }, //west
      { id: 2, position: { x: 600, y: 550 }, initialPosition: {x: 600, y: 550 }, totalNumCars: 0 }    //east
  ];
  pedestrians: Pedestrian[] = [
      { id: 0, position: { x: 740, y: 75 }, initialPosition: { x: 740, y: 75 }, totalNumPeds: 0 },   //west
      { id: 1, position: { x: 1125, y: 75 }, initialPosition: { x: 1125, y: 75 }, totalNumPeds: 0 }  //east
  ];

  currentState: TrafficState = 'base';
  isLoadingModel = true;
  statusMessage = 'Loading AI model...';
  carsL1_West = 0;
  carsL2_East = 0;
  carsL3_North = 0;
  pedsW1_West = 0;
  pedsW2_East = 0;
  pedsW3_South = 0;
  waitTimeL1_L2 = 0;
  waitTimeL3 = 0;
  waitTimeW1_W2 = 0;

  private model: tf.LayersModel | undefined;
  private predictionInterval: any;
  private stateTimer: any;
  private animationFrameId: number | undefined;
  private readonly SCALER_MEAN = [6.994502748625687, 7.011694152923538, 1.9861069465267367, 3.0368815592203897, 3.049775112443778, 0.0, 14.593003498250875, 14.50064967516242, 14.522038980509745];
  private readonly SCALER_SCALE = [4.325195499288317, 4.307050944032767, 1.4159464800406658, 1.985287985726915, 2.002776982486944, 1.0, 8.699760232927833, 8.675925814190926, 8.68162160910034];
  private app: FirebaseApp;
  private db: Database;
  readonly firebaseConfig = {
      apiKey: "<env-var>",
      authDomain: "<env-var>",
      databaseURL: "<env-var>",
      projectId: "<env-var>",
      storageBucket: "<env-var>",
      messagingSenderId: "<env-var>",
      appId: "<env-var>",
      measurementId: "<env-var>"
  };

  constructor() {
    this.app = initializeApp(this.firebaseConfig);
    this.db = getDatabase(this.app);
  }

  ngOnInit(): void {
    this.loadModel().then(() => {
      this.pollTrafficData(); // initial call
      setInterval(() => this.pollTrafficData(), 5000); // poll every 5s
    });
  }

  ngAfterViewInit(): void {
    this.updatePositions();
  }

  ngOnDestroy(): void {
    if (this.predictionInterval) clearInterval(this.predictionInterval);
    if (this.stateTimer) clearTimeout(this.stateTimer);
    if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
  }

  private async loadModel(): Promise<void> {
    try {
      this.model = await tf.loadLayersModel('assets/tfjs_model/model.json');
      this.isLoadingModel = false;
      this.statusMessage = 'AI Model Loaded. Listening for traffic data...';
      this.setBaseState(); // Initialize with base state
    } catch (error) {
      this.isLoadingModel = false;
      this.statusMessage = 'Error loading AI model. Please check console.';
      console.error('Model loading error:', error);
    }
  }

  private async pollTrafficData(): Promise<void> {
    const trafficDataRef = ref(this.db, 'trafficData/');
    try {
      const snapshot = await get(trafficDataRef);
      const data = snapshot.val();
      if (data) {
        console.log('Polled data:', data);
        this.statusMessage = 'Polled data received. AI is active.';
        this.carsL1_West = data.carsL1_West || 0;
        this.carsL2_East = data.carsL2_East || 0;
        this.carsL3_North = data.carsL3_North || 0;
        this.pedsW1_West = data.pedsW1_West || 0;
        this.pedsW2_East = data.pedsW2_East || 0;
        this.pedsW3_South = data.pedsW3_South || 0;
        this.waitTimeL1_L2 = data.waitTimeL1_L2 || 0;
        this.waitTimeL3 = data.waitTimeL3 || 0;
        this.waitTimeW1_W2 = data.waitTimeW1_W2 || 0;
        //TBD: set wait times to zero depending on the states
        this.updateCarsAndPedsBasedOnData();
        this.runPrediction();
      } else {
        this.statusMessage = 'ERROR: AI active, but no data found at Firebase path';
      }
    } catch (err) {
      console.error('Error polling traffic data:', err);
      this.statusMessage = 'ERROR: Failed to poll Firebase data';
    }
  }

  private runPrediction(): void {
    if (!this.model || this.isLoadingModel) return;

    const inputData = [
        this.carsL1_West, this.carsL2_East, this.carsL3_North,
        this.pedsW1_West, this.pedsW2_East, this.pedsW3_South,
        this.waitTimeL1_L2, this.waitTimeL3, this.waitTimeW1_W2
    ];

    const scaledData = inputData.map((val, i) => (val - this.SCALER_MEAN[i]) / this.SCALER_SCALE[i]);
    const inputTensor = tf.tensor2d([scaledData]);
    console.log('running prediction with input:');
    this.statusMessage = 'Running AI prediction...';
    const prediction = this.model.predict(inputTensor) as tf.Tensor;
    const predictedStateIndex = prediction.argMax(-1).dataSync()[0];
    inputTensor.dispose();
    prediction.dispose();

    switch (predictedStateIndex) {
      case 0:
        if (this.currentState !== 'base') {
          this.setYellowTransition(() => this.setBaseState());
        }
        break;
      case 1:
        if (this.currentState !== 'car_only') {
          this.setYellowTransition(() => this.setCarOnlyState());
        }
        break;
      case 2:
        if (this.currentState !== 'pedestrian') {
          this.setYellowTransition(() => this.setPedestrianState());
        }
        break;
      default:
        this.setYellowTransition(() => this.setBaseState());
        break;
    }
  }

  private setBaseState(): void {
    clearTimeout(this.stateTimer);
    this.stateTimer = null;
    this.currentState = 'base';
    this.statusMessage = 'State: Base (East/West traffic flow)';

    this.trafficLights[0].activeLight = 'red';
    this.trafficLights[1].activeLight = 'green';
    this.trafficLights[2].activeLight = 'green';

    this.pedestrianSigns[0].activeLight = 'dontWalk';
    this.pedestrianSigns[1].activeLight = 'dontWalk';
    this.pedestrianSigns[2].activeLight = 'walk';
  }

  private setCarOnlyState(): void {
    clearTimeout(this.stateTimer);
    this.currentState = 'car_only';
    
    const duration = this.carsL3_North > 3 ? 10000 : 5000; // 10 or 5 seconds
    this.statusMessage = `State: Car Only (North flow) for ${duration / 1000}s`;

    this.trafficLights[0].activeLight = 'green';
    this.trafficLights[1].activeLight = 'red';
    this.trafficLights[2].activeLight = 'red';

    this.pedestrianSigns.forEach(p => p.activeLight = 'dontWalk');

    this.stateTimer = setTimeout(() => this.setBaseState(), duration);
  }

  private setPedestrianState(): void {
    clearTimeout(this.stateTimer);
    this.currentState = 'pedestrian';
    
    const totalPeds = this.pedsW1_West + this.pedsW2_East;
    const duration = totalPeds > 5 ? 25000 : 15000; // 25 or 15 seconds
    this.statusMessage = `State: Pedestrian Crossing (East/West) for ${duration / 1000}s`;

    this.trafficLights[0].activeLight = 'green';
    this.trafficLights[1].activeLight = 'red';
    this.trafficLights[2].activeLight = 'red';

    this.pedestrianSigns[0].activeLight = 'walk';
    this.pedestrianSigns[1].activeLight = 'walk';
    this.pedestrianSigns[2].activeLight = 'dontWalk';

    this.stateTimer = setTimeout(() => this.setBaseState(), duration);
  }

  private updateCarsAndPedsBasedOnData(): void {
    this.cars[0].totalNumCars = this.carsL3_North;
    this.cars[1].totalNumCars = this.carsL1_West;
    this.cars[2].totalNumCars = this.carsL2_East;
    this.pedestrians[0].totalNumPeds = this.pedsW1_West;
    this.pedestrians[1].totalNumPeds = this.pedsW2_East;
  }

  private setYellowTransition(nextStateSetter: () => void): void {
    this.trafficLights.forEach(light => {
      if (light.activeLight === 'green') {
        light.activeLight = 'yellow';
      }
    });

    this.statusMessage = 'Transitioning: Yellow phase...';
    this.resetAll();

    this.stateTimer = setTimeout(() => {
      nextStateSetter();
    }, 3000);
  }


  updatePositions(): void {
    this.cars.forEach(car => {
        if (car.id === 0 && this.trafficLights[0].activeLight == 'green') {
          if (car.position.y < 400) {
            car.position.x -= 1.5;
          }
          if (car.position.y > 200) {
            car.position.y -= 1.5;
          }
          if (car.position.x < 500) {
            car.position.x = car.initialPosition.x;
            car.position.y = car.initialPosition.y;
          }
        }else if (car.id === 1 && this.trafficLights[1].activeLight == 'green') {
          car.position.x -= 1.5;
          if (car.position.x < 600) {
            car.position.x = car.initialPosition.x;
          }
        }else if (car.id === 2 && this.trafficLights[2].activeLight == 'green') {
          car.position.x += 1.5;
          if (car.position.x > 1300) {
            car.position.x = car.initialPosition.x;
          }
        }
    });

    this.pedestrians.forEach(pedestrian => {
      if (pedestrian.id === 0 && this.pedestrianSigns[0].activeLight === 'walk' && pedestrian.totalNumPeds > 0) {
          pedestrian.position.y += 1.5;
          if (pedestrian.position.y > 800) {
            pedestrian.position.y = pedestrian.initialPosition.y;
          }
      } else if (pedestrian.id === 1 && this.pedestrianSigns[1].activeLight === 'walk' && pedestrian.totalNumPeds > 0) {
          pedestrian.position.y += 1.5;
          if (pedestrian.position.y > 800) {
            pedestrian.position.y = pedestrian.initialPosition.y;
        }
      }
    });

    this.animationFrameId = requestAnimationFrame(() => this.updatePositions());
  }

  resetAll(): void {
    this.cars.forEach(car => {
      car.position.x = car.initialPosition.x;
      car.position.y = car.initialPosition.y;
    });
    this.pedestrians.forEach(pedestrian => {
      pedestrian.position.x = pedestrian.initialPosition.x;
      pedestrian.position.y = pedestrian.initialPosition.y;
    });
  }
}
