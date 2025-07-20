import { ComponentFixture, TestBed } from '@angular/core/testing';

import { StopLightComponent } from './stop-light.component';

describe('StopLightComponent', () => {
  let component: StopLightComponent;
  let fixture: ComponentFixture<StopLightComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ StopLightComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(StopLightComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
